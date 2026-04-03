import argparse
import datetime
import multiprocessing as mp
import os
import threading
import time
import asyncio


def monte_carlo_partial(chunk_size: int) -> int:
    """Executa uma parte da simulação Monte Carlo (chunk)."""
    import random
    random.seed(os.getpid() + int(time.time() * 1000) % 10000)
    hits = 0
    for _ in range(chunk_size):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x * x + y * y <= 1.0:
            hits += 1
    return hits


def serial_monte_carlo(total_samples: int) -> int:
    """Versão serial."""
    return monte_carlo_partial(total_samples)


def parallel_monte_carlo(
    total_samples: int,
    num_workers: int,
    n_tasks: int,
    chunk_size: int,
    logfile: str,
) -> int:
    """Versão paralela com multiprocessing.Pool + logging contínuo."""
    def write_log(progress: float, hits: int, total: int, pid: int = None):
        if pid is None:
            pid = os.getpid()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pi_est = 4 * hits / total if total > 0 else 0
        with open(logfile, "a") as f:
            f.write(
                f"[{timestamp}] PID:{pid} Progresso:{progress:.1f}% "
                f"Hits:{hits}/{total} π_est:{pi_est:.6f}\n"
            )

    open(logfile, "w").close()

    with mp.Pool(processes=num_workers) as pool:
        chunks = [chunk_size] * n_tasks
        results = pool.imap_unordered(monte_carlo_partial, chunks)

        hits_total = 0
        completed = 0

        for hits in results:
            hits_total += hits
            completed += 1
            progress = (completed / n_tasks) * 100
            write_log(progress, hits_total, total_samples)

    return hits_total


async def monitoring_task(task_id: int):
    """Tarefa assíncrona de monitoramento."""
    try:
        while True:
            await asyncio.sleep(2.0)
    except asyncio.CancelledError:
        pass


def start_async_monitoring():
    """Inicia 20 tarefas assíncronas em thread separada (versão corrigida)."""
    def run_event_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        tasks = [loop.create_task(monitoring_task(i)) for i in range(20)]

        try:
            loop.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            loop.close()

    thread = threading.Thread(
        target=run_event_loop,
        daemon=True,
        name="AsyncMonitor"
    )
    thread.start()
    print("20 tarefas assíncronas (asyncio) iniciadas em background.")
    return thread


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aproximação de π por Monte Carlo - Serial + Paralela + Async"
    )
    parser.add_argument("--n_tasks", type=int, default=100,
                        help="Número de blocos (n_tasks)")
    parser.add_argument("--chunk_size", type=int, default=100_000,
                        help="Tamanho de cada bloco")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count(),
                        help="Número de workers (padrão = CPUs disponíveis)")
    parser.add_argument("--logfile", default="montecarlo.log",
                        help="Arquivo de log")

    args = parser.parse_args()

    total_samples = args.n_tasks * args.chunk_size

    print(f"Monte Carlo - Total de amostras: {total_samples:,}")
    print(f"Workers: {args.num_workers} | Blocos: {args.n_tasks} | Chunk: {args.chunk_size}\n")

    # Serial
    print("=== EXECUTANDO VERSÃO SERIAL ===")
    start_time = time.time()
    hits_serial = serial_monte_carlo(total_samples)
    pi_serial = 4 * hits_serial / total_samples
    serial_time = time.time() - start_time

    print(f"Tempo total (wall clock): {serial_time:.2f} segundos")
    print(f"π estimado (serial): {pi_serial:.6f}\n")

    # Paralela
    print("=== EXECUTANDO VERSÃO PARALELA ===")
    async_thread = start_async_monitoring()

    start_time = time.time()
    hits_parallel = parallel_monte_carlo(
        total_samples,
        args.num_workers,
        args.n_tasks,
        args.chunk_size,
        args.logfile,
    )
    pi_parallel = 4 * hits_parallel / total_samples
    parallel_time = time.time() - start_time

    print(f"Tempo total (wall clock): {parallel_time:.2f} segundos")
    print(f"π estimado (paralelo): {pi_parallel:.6f}\n")

    # Comparacao
    print("=== COMPARAÇÃO FINAL ===")
    print(f"Serial   → {serial_time:6.2f}s | π = {pi_serial:.6f}")
    print(f"Paralelo → {parallel_time:6.2f}s | π = {pi_parallel:.6f}")
    print(f"Speedup  → {serial_time / parallel_time:.2f}x")
    print(f"\nLog contínuo gerado em: {args.logfile}")
    print("20 tarefas assíncronas (asyncio) rodando em background.")