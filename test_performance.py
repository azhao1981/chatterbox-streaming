import time
import torch
import torchaudio as ta
from pathlib import Path
from chatterbox.tts import ChatterboxTTS


def test_performance(model, text, rounds=3):
    """同时测试首包延迟和RTF"""
    results = []

    for i in range(rounds):
        start_time = time.time()
        audio_chunks = []
        first_chunk_time = None

        for j, (audio_chunk, metrics) in enumerate(model.generate_stream(text)):
            audio_chunks.append(audio_chunk)

            if j == 0 and first_chunk_time is None:
                first_chunk_time = time.time()
                first_chunk_latency = first_chunk_time - start_time

        total_time = time.time() - start_time
        total_audio = torch.cat(audio_chunks, dim=-1)
        audio_duration = total_audio.shape[-1] / model.sr
        rtf = audio_duration / total_time

        # 保存音频文件
        ta.save(f"test_round_{i+1}.wav", total_audio, model.sr)

        results.append({
            'first_chunk_latency': first_chunk_latency,
            'rtf': rtf,
            'audio_duration': audio_duration,
            'total_time': total_time
        })

        print(f"Round {i+1}: 首包延迟 = {first_chunk_latency:.3f}s, RTF = {rtf:.3f}, 已保存 test_round_{i+1}.wav")

    # 计算平均值
    avg_latency = sum(r['first_chunk_latency'] for r in results) / len(results)
    avg_rtf = sum(r['rtf'] for r in results) / len(results)

    print(f"平均首包延迟: {avg_latency:.3f}s")
    print(f"平均RTF: {avg_rtf:.3f}")

    return results


def main():
    # 设备检测
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"使用设备: {device}")

    # 加载模型
    model_path = Path("./ResembleAI/chatterbox")
    model = ChatterboxTTS.from_local(model_path, device=device)

    # 测试文本
    test_text = "Hello world, this is a simple test for text to speech performance."

    print("=" * 50)
    print("性能测试 (首包延迟 + RTF)")
    print("=" * 50)
    test_performance(model, test_text)


if __name__ == "__main__":
    main()