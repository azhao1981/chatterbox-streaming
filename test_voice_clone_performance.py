import time
import torch
import torchaudio as ta
from pathlib import Path
from chatterbox.tts import ChatterboxTTS


def test_clone_performance(model, text, reference_audio_path, rounds=3):
    """测试克隆音性能 (首包延迟 + RTF)"""
    results = []

    # 加载参考音频
    reference_audio, ref_sr = ta.load(reference_audio_path)
    if ref_sr != model.sr:
        # 重采样到模型采样率
        reference_audio = ta.transforms.Resample(ref_sr, model.sr)(reference_audio)

    for i in range(rounds):
        start_time = time.time()
        audio_chunks = []
        first_chunk_time = None

        for j, (audio_chunk, metrics) in enumerate(model.generate_stream(
            text=text,
            audio_prompt_path=reference_audio_path
        )):
            audio_chunks.append(audio_chunk)

            if j == 0 and first_chunk_time is None:
                first_chunk_time = time.time()
                first_chunk_latency = first_chunk_time - start_time

        total_time = time.time() - start_time
        total_audio = torch.cat(audio_chunks, dim=-1)
        audio_duration = total_audio.shape[-1] / model.sr
        rtf = total_time / audio_duration 

        # 保存音频文件
        ta.save(f"clone_test_round_{i+1}.wav", total_audio, model.sr)

        results.append({
            'first_chunk_latency': first_chunk_latency,
            'rtf': rtf,
            'audio_duration': audio_duration,
            'total_time': total_time
        })

        print(f"Round {i+1}: 首包延迟 = {first_chunk_latency:.3f}s, RTF = {rtf:.3f}, 已保存 clone_test_round_{i+1}.wav")

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

    # 测试文本和参考音频
    test_text = "Hello world, this is a voice cloning performance test., 中文如何"
    reference_audio_path = "clonetts.mp3"  # 需要用户提供参考音频文件

    # 检查参考音频是否存在
    if not Path(reference_audio_path).exists():
        print(f"错误: 找不到参考音频文件 {reference_audio_path}")
        print("请准备一个参考音频文件 (3-10秒的语音) 并命名为 clonetts.mp3")
        return

    print("=" * 50)
    print("克隆音性能测试 (首包延迟 + RTF)")
    print(f"参考音频: {reference_audio_path}")
    print("=" * 50)
    test_clone_performance(model, test_text, reference_audio_path)


if __name__ == "__main__":
    main()