import pyttsx3
import time
import logging

logger = logging.getLogger(__name__)


def init_tts() -> pyttsx3.Engine | None:
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # 设置语速
        engine.setProperty('volume', 1.0)  # 设置音量
        voices = engine.getProperty('voices')
        # 存储中文和英文语音
        chinese_voice = None
        english_voice = None
        for voice in voices:
            voice_id_lower = voice.id.lower()
            if 'zh' in voice_id_lower or 'chinese' in voice_id_lower:
                chinese_voice = voice.id
            elif 'en' in voice_id_lower or 'english' in voice_id_lower:
                english_voice = voice.id
        # 返回找到的语音
        return engine, chinese_voice, english_voice
    except Exception as e:
        logger.info(f"初始化语音合成引擎失败: {e}")
        return None, None, None


def process_tts_detection(result, tts_enable, tts_duration, tts_interval, tts_engine, tts_state, tts_text_cn,
                          tts_text_en, chinese_voice, english_voice):
    if not tts_enable or not tts_engine:
        logger.info("语音合成功能未启用或引擎未初始化")
        return

    classes = result.boxes.cls.cpu().numpy() if result.boxes else []
    has_face = 0 in classes
    current_time = time.time()
    in_cooldown = tts_state.get('last_tts_time') and (current_time - tts_state.get('last_tts_time') < tts_interval)

    if not in_cooldown:
        if has_face:
            if tts_state.get('no_mask_start_time') is None:
                tts_state['no_mask_start_time'] = current_time
                logger.info(f"检测到未戴口罩,开始计时: {tts_state.get('no_mask_start_time')}")
            elif current_time - tts_state.get('no_mask_start_time') >= tts_duration:
                logger.info(f"未戴口罩时间超过阈值({tts_duration}秒),触发语音提醒")
                try:
                    # 中文语音提醒
                    if chinese_voice:
                        tts_engine.setProperty('voice', chinese_voice)
                        tts_engine.say(tts_text_cn)

                    # 英文语音提醒
                    if english_voice:
                        tts_engine.setProperty('voice', english_voice)
                        tts_engine.say(tts_text_en)

                    tts_engine.runAndWait()
                    logger.info(f"语音提醒完成: {tts_text_cn} | {tts_text_en}")
                    tts_state['last_tts_time'] = current_time
                    tts_state['no_mask_start_time'] = None
                except Exception as e:
                    logger.error(f"语音提醒失败: {e}")
    else:
        if tts_state.get('no_mask_start_time') is not None:
            logger.info("已戴上口罩,重置计时状态")
            tts_state['no_mask_start_time'] = None
