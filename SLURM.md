<!-- IMPORTANTE PARA estimate_time.py
     GPU Speed: fator de velocidade relativo ao laptop de referência (CPU pura, treino).
     CPU Speed: fator single-thread relativo ao laptop de referência (operações HE).
     Quanto maior, mais rápida a máquina.  Atualize após rodar experimentos reais
     com o comando: python3 experiments/estimate_time.py --calibrate <log_dir>
     A linha "local" representa esta máquina e DEVE ter GPU_Speed=1.0 e CPU_Speed=1.0.

     Nota sobre partições mistas (l40s):
       dl-01,02,03 → AMD Threadripper PRO 7975WX (32 cores, Zen 4, ~5 GHz boost)
       dl-05       → Intel Xeon Silver 4410Y     (12 cores, Sapphire Rapids)
       CPU_Speed usa o valor conservador (pior nó da partição, dl-05).

     (*) Modelos marcados com asterisco foram inferidos via scontrol (nós ocupados
         durante coleta; confirmar com o comando abaixo assim que o nó liberar:
         srun -p <partição> -w <nó> lscpu
         e buscar o campo "Model name")
-->

| Partição | Nós | GPUs | Tempo máx | GPU Speed | CPU Speed | CPU | Cores | RAM | Nota |
|---|---|---|---|---|---|---|---|---|---|
| **local** | — | CPU (12 threads) | — | 1.0 | 1.0 | Consumer CPU | 6c/12t | — | Referência: resnet20/CIFAR10 = 163 s/round |
| **l40s** (default) | dl-01, 02, 03, 05 | L40S 46GB | 4 dias | 12.0 | 0.8 | dl-01–03: AMD Threadripper PRO 7975WX / dl-05: Intel Xeon Silver 4410Y | 32c (dl-01–03) / 12c (dl-05) | — | CPU Speed usa dl-05 (conservador); Threadripper com alta largura de banda para L40S |
| **h100** | dl-04 | H100 NVL | 2 dias | 25.0 | 1.3 | AMD Threadripper PRO (presumido*) | 32c | — | Nó mais potente; só 2 GPUs disponíveis |
| **a5000** | dl-06 | RTX A5000 | 3 dias | 10.0 | 0.7 | Intel Xeon Gold 61xx/62xx* | 20c/socket × 2 sockets (80t total) | 1 TB | Ideal para datasets que não cabem em RAM convencional |
| **rtx8000** | dl-07, 08 | Quadro RTX 8000 48GB | 5 dias | 7.0 | 0.7 | Intel Xeon Gold 61xx/62xx* | 20c/socket × 2 sockets (80t total) | 1 TB | 8 GPUs/nó; maior tempo máx permitido |
| **p5000** | dl-09 | Quadro P5000 | 3 dias | 4.0 | 0.6 | Intel Xeon Silver/Gold* | 16c | — | GPUs Pascal (arquitetura antiga); geralmente menos concorrida |
| **rtx5000** | dl-10 | RTX 5000 | 3 dias | 6.0 | 0.7 | Intel Xeon Gold* | 18c | — | Boa para testes rápidos e validação de código |

---

### Notas de Topologia

1. **Modelos marcados com (\*)** foram inferidos via `scontrol` (nós ocupados durante a coleta).
   Para confirmar, execute `srun -p <partição> -w <nó> lscpu` e procure o campo "Model name".

2. **Threads totais** nos nós de 20 cores/socket (dl-06, 07, 08):
   2 sockets × 20 cores × 2 threads (HT) = **80 threads** por nó.

3. **Threadripper PRO** (dl-01 a 04): arquitetura Zen 4 com alta largura de banda de
   memória — ideal para alimentar GPUs de nova geração (L40S, H100) sem gargalo de PCIe.

4. **Memória de 1 TB** exclusiva nos nós `a5000` (dl-06) e `rtx8000` (dl-07, 08) —
   únicos indicados para datasets que excedam ~128 GB de RAM.

5. **Partição l40s é mista**: jobs podem cair em qualquer nó (dl-01–03 ou dl-05).
   O `CPU Speed=0.8` usa o Xeon Silver 4410Y (dl-05) como pior caso; se o job cair
   num Threadripper (dl-01–03), o overhead HE será ~30% menor que o estimado.
