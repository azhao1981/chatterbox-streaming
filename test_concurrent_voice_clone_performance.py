import time
import torch
import torchaudio as ta
import threading
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from chatterbox.tts import ChatterboxTTS


def test_single_clone_request(model, request_id, text, reference_audio_path, lock):
    """单个克隆音请求的测试函数（带锁保护）"""
    start_time = time.time()
    audio_chunks = []
    first_chunk_time = None

    # 使用锁保护模型推理
    with lock:
        for j, (audio_chunk, metrics) in enumerate(model.generate_stream(
            text=text,
            audio_prompt_path=reference_audio_path
        )):
            audio_chunks.append(audio_chunk)

            if j == 0 and first_chunk_time is None:
                first_chunk_time = time.time()

    total_time = time.time() - start_time
    total_audio = torch.cat(audio_chunks, dim=-1)
    audio_duration = total_audio.shape[-1] / model.sr
    rtf = total_time / audio_duration
    first_chunk_latency = first_chunk_time - start_time if first_chunk_time else total_time

    # 保存音频文件
    output_file = f"concurrent_clone_req_{request_id}.wav"
    ta.save(output_file, total_audio, model.sr)

    return {
        'request_id': request_id,
        'first_chunk_latency': first_chunk_latency,
        'rtf': rtf,
        'audio_duration': audio_duration,
        'total_time': total_time,
        'output_file': output_file
    }


def warmup_clone_model(model, reference_audio_path):
    """克隆音模型预热 - 发送一个简单请求，不计入统计"""
    print("正在预热克隆音模型...")
    warmup_text = "Warm up"

    warmup_start = time.time()
    with threading.Lock():
        audio_chunks = []
        for audio_chunk, metrics in model.generate_stream(
            text=warmup_text,
            audio_prompt_path=reference_audio_path
        ):
            audio_chunks.append(audio_chunk)

    warmup_time = time.time() - warmup_start
    print(f"克隆音模型预热完成，耗时: {warmup_time:.3f}s")
    print("-" * 60)


def test_concurrent_clone_performance(model, text, reference_audio_path, concurrency=3, rounds=1):
    """并发克隆音性能测试"""
    # 先进行模型预热
    warmup_clone_model(model, reference_audio_path)

    print(f"开始并发克隆音测试: {concurrency}个并发请求, {rounds}轮测试")
    print(f"测试文本: {text}")
    print(f"参考音频: {reference_audio_path}")
    print("=" * 60)

    # 创建线程锁保护模型访问
    model_lock = threading.Lock()
    all_results = []

    for round_num in range(rounds):
        print(f"\n--- 第 {round_num + 1} 轮测试 ---")
        round_start_time = time.time()
        round_results = []

        # 使用线程池执行并发请求
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            # 提交所有任务
            futures = []
            for i in range(concurrency):
                future = executor.submit(
                    test_single_clone_request,
                    model,
                    f"r{round_num+1}_req{i+1}",
                    text,
                    reference_audio_path,
                    model_lock
                )
                futures.append(future)

            # 收集结果
            for future in as_completed(futures):
                result = future.result()
                round_results.append(result)
                print(f"请求 {result['request_id']}: "
                      f"首包延迟={result['first_chunk_latency']:.3f}s, "
                      f"RTF={result['rtf']:.3f}, "
                      f"总时间={result['total_time']:.3f}s")

        round_total_time = time.time() - round_start_time
        round_results.sort(key=lambda x: x['request_id'])  # 按请求ID排序

        # 计算本轮统计
        avg_latency = sum(r['first_chunk_latency'] for r in round_results) / len(round_results)
        avg_rtf = sum(r['rtf'] for r in round_results) / len(round_results)
        max_latency = max(r['first_chunk_latency'] for r in round_results)
        min_latency = min(r['first_chunk_latency'] for r in round_results)

        print(f"本轮统计: 平均首包延迟={avg_latency:.3f}s, "
              f"平均RTF={avg_rtf:.3f}, "
              f"最大延迟={max_latency:.3f}s, "
              f"最小延迟={min_latency:.3f}s, "
              f"轮次总时间={round_total_time:.3f}s")

        all_results.extend(round_results)

    # 总体统计
    print(f"\n{'='*60}")
    print("总体统计:")
    print(f"总请求数: {len(all_results)}")
    print(f"总并发数: {concurrency}")
    print(f"测试轮数: {rounds}")

    avg_latency = sum(r['first_chunk_latency'] for r in all_results) / len(all_results)
    avg_rtf = sum(r['rtf'] for r in all_results) / len(all_results)
    max_latency = max(r['first_chunk_latency'] for r in all_results)
    min_latency = min(r['first_chunk_latency'] for r in all_results)

    print(f"平均首包延迟: {avg_latency:.3f}s")
    print(f"平均RTF: {avg_rtf:.3f}")
    print(f"最大首包延迟: {max_latency:.3f}s")
    print(f"最小首包延迟: {min_latency:.3f}s")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Chatterbox并发克隆音性能测试')
    parser.add_argument('--concurrency', type=int, default=3,
                       help='并发请求数 (默认: 3)')
    parser.add_argument('--rounds', type=int, default=1,
                       help='测试轮数 (默认: 1)')
    parser.add_argument('--text', type=str,
                       default="Hello world, this is a concurrent voice cloning performance test.",
                       help='测试文本')
    parser.add_argument('--reference', type=str, default="clonetts.mp3",
                       help='参考音频文件路径 (默认: clonetts.mp3)')

    args = parser.parse_args()

    # 检查参考音频是否存在
    if not Path(args.reference).exists():
        print(f"错误: 找不到参考音频文件 {args.reference}")
        print("请准备一个参考音频文件 (3-10秒的语音) 并使用 --reference 参数指定")
        return

    # 设备检测
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"使用设备: {device}")

    # 加载单一模型实例
    model_path = Path("./ResembleAI/chatterbox")
    print("正在加载模型...")
    model = ChatterboxTTS.from_local(model_path, device=device)
    print("模型加载完成")

    print("=" * 60)
    print(f"Chatterbox并发克隆音性能测试")
    print(f"并发数: {args.concurrency}, 轮数: {args.rounds}")
    print(f"参考音频: {args.reference}")
    print("=" * 60)

    # 执行并发克隆音测试
    test_concurrent_clone_performance(model, args.text, args.reference, args.concurrency, args.rounds)


if __name__ == "__main__":
    main()