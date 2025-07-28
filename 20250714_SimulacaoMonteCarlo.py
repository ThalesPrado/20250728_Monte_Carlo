import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from scipy.stats import norm

st.title("📘 Introdução Teórica + Simulação Monte Carlo")

st.header("🎯 Intuição Inicial: Estimando π com Monte Carlo")

st.markdown(r"""
Monte Carlo é uma técnica que utiliza números aleatórios para simular fenômenos complexos.  
Vamos começar estimando o valor de $π$ com uma ideia clássica.

Imagine um **quadrado de lado 1** com um **quarto de círculo** dentro dele.  
Ao jogar "dardos aleatórios" nesse quadrado, a proporção dos que caem no círculo é:

$$
\frac{\pi}{4} \Rightarrow \pi \approx 4 \cdot \frac{n_{\text{dentro}}}{n_{\text{total}}}
$$
""")

np.random.seed(10)
n = 10000
nc = 0
pi_values = []
x_in, y_in = [], []
x_out, y_out = [], []

for j in range(n):
    x, y = np.random.random(2)
    if x**2 + y**2 <= 1:
        nc += 1
        x_in.append(x)
        y_in.append(y)
    else:
        x_out.append(x)
        y_out.append(y)
    if (j+1) % 10 == 0:
        pi_values.append(4 * nc / (j+1))

fig2, ax2 = plt.subplots()
ax2.scatter(x_in, y_in, s=1, label='Dentro do círculo', color='blue')
ax2.scatter(x_out, y_out, s=1, label='Fora do círculo', color='gray')
circle = plt.Circle((0, 0), 1, color='red', fill=False)
ax2.add_artist(circle)
ax2.set_aspect('equal')
ax2.set_title('🎯 Disposição dos Pontos')
ax2.legend()
st.pyplot(fig2)

fig1, ax1 = plt.subplots()
ax1.plot(range(10, n+1, 10), pi_values, label='Estimativa de π')
ax1.axhline(np.pi, color='r', linestyle='--', label='π real')
ax1.set_title('📈 Convergência de π')
ax1.set_xlabel('Simulações')
ax1.set_ylabel('Estimativa')
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

st.markdown("""
# 🔢 Geradores Pseudoaleatórios (PRNGs)
Computadores usam algoritmos determinísticos para gerar números "aleatórios".  
Esses PRNGs simulam aleatoriedade, e sua qualidade impacta diretamente a precisão da simulação de Monte Carlo.

## ⚙️ Métodos Clássicos
- **LCG**: fórmula simples, mas pode gerar padrões.
- **von Neumann**: histórico, não confiável.
- **Mersenne Twister (NumPy)**: padrão moderno e confiável.
""")

n_test = st.slider("Tamanho da amostra para teste estatístico", 100, 5000, 1000, key="prng_teste")
np_vals = np.random.random(n_test)

fig, ax = plt.subplots()
lag_plot(pd.Series(np_vals), ax=ax)
ax.set_title("Lag Plot - NumPy (Mersenne Twister)")
st.pyplot(fig)

fig_acf, ax_acf = plt.subplots()
plot_acf(np_vals, lags=30, ax=ax_acf)
ax_acf.set_title("ACF - NumPy")
st.pyplot(fig_acf)

acf_np = acf(np_vals, nlags=1)[1]
st.markdown(f"""
### Coeficiente de Autocorrelação (Lag 1)
- NumPy: `r = {acf_np:.4f}`
- Valores próximos de 0 indicam falta de autocorrelação logo os resultados são independentes entre si e isso é exatamente o que esperamos de uma simulação Monte Carlo bem feita.""")

def runs_test(series):
    median = np.median(series)
    runs, n1, n2 = 0, 0, 0
    prev = None
    for val in series:
        curr = 'A' if val >= median else 'B'
        if curr != prev:
            runs += 1
        if curr == 'A':
            n1 += 1
        else:
            n2 += 1
        prev = curr
    expected = (2 * n1 * n2) / (n1 + n2) + 1
    std = np.sqrt((2 * n1 * n2 * (2*n1*n2 - n1 - n2)) / (((n1 + n2)**2) * (n1 + n2 - 1)))
    z = (runs - expected) / std if std > 0 else 0
    p = 2 * (1 - norm.cdf(abs(z)))
    return runs, expected, z, p

runs_np, exp_np, z_np, p_np = runs_test(np_vals)

st.markdown(f"""
### Runs Test (Teste de Aleatoriedade)
- Runs observados: `{runs_np}`
- Esperado: `{exp_np:.2f}`
- Estatística z: `{z_np:.2f}`  
- p-valor: `{p_np:.4f}`

#### Hipóteses:
- H₀: Os dados são aleatórios.
- H₁: Os dados não são aleatórios.

Como o p-valor é alto, não rejeitamos H₀.  
A sequência gerada pelo Mersenne Twister é estatisticamente aleatória.
""")

st.markdown("""
## Conclusão Final
- O NumPy/Mersenne Twister passou em todos os testes estatísticos.
- Os gráficos confirmam ausência de padrões, autocorrelação e viés.
- Ele é, portanto, adequado para simulações de Monte Carlo.
""")

st.markdown(r"""
---
## Visão Geral do Projeto
Este aplicativo demonstra o uso da **Simulação de Monte Carlo** para calcular o preço unitário de uma barra de proteína considerando a incerteza nos custos de matérias-primas.

Temas chaves da simulação de Monte Carlo são:
            
### 1. 🧮 Lei dos Grandes Números (LLN)
**Intuição:** Quanto mais simulações fazemos, mais estável e confiável será a nossa estimativa de preço unitário médio.
            
**Exemplo:**  
Se jogar uma moeda 10 vezes, pode dar 7 caras. Mas se jogar 10.000 vezes, o número de caras se aproxima de 50%.
            
**Lei dos Grandes Números (LLN):**
   À medida que $N \to \infty$:
   $$
   \lim_{N \to \infty} \hat{E}[X] = E[X]
   $$
   Ou seja, a média das simulações converge para o valor esperado verdadeiro com probabilidade 1 (convergência quase certa).
            
### 2. 🔄 Teorema Central do Limite (CLT)
**Intuição:** Mesmo que cada preço simulado siga uma distribuição diferente (normal ou triangular), a **média dos preços simulados** tende a formar uma curva de sino (distribuição normal).

**Exemplo:**  
A distribuição de preços de proteína, cacau e açúcar pode variar, mas a média do preço da barra simula uma curva suave parecida com uma Gaussiana.

### 3. ⏳ Ergodicidade
**Intuição:** Simular uma longa sequência de cenários é equivalente a observar todos os cenários possíveis.

**Exemplo:**  
Ao simular 10.000 cenários de custos, podemos entender o comportamento geral do preço mesmo sem testar **todas** as combinações possíveis.

### 4. 📉 Desigualdade de Chebyshev
**Intuição:** A maioria das simulações fica perto da média. Simulações muito distantes são raras.

        
$$
P\left(|X - \mu| \geq k\sigma\right) \leq \frac{1}{k^2}
$$

Isso também implica que a **probabilidade de estar dentro de $k$ desvios padrão** da média é **pelo menos**:

$$
P\left(|X - \mu| < k\sigma\right) \geq 1 - \frac{1}{k^2}
$$
            
**Exemplo:**  
Imaginemos que trabalhamos em uma loja de roupas e queremos analisar o preço médio de venda das camisetas nos últimos 6 meses para ajudar a definir a faixa de preço ideal para os próximos lançamentos.
            
Com uma média dos preços de camisetas vendidas de 𝜇 = 60 e desvio padrão = 10. Temos P([|X - 𝜇| >= k*]) >= 1 - (1 / k^2).
O que queremos responder aqui é qual a probabilidade de que X esteja a uma distância maior ou igual a k desvios padrão da média.

Logo temos, P([|X - 60|] < 20) >= 1 - 1/4.
            
            P([|X - 60|] < 20) >= 0,75.
            
            -20 < |X - 60| < 20

            -20 + 60 < | X - 60 + 60 | < 20 + 60

            40 < |X| < 80

            P(40 < |X| < 80) >= 0,75

Pelo menos 75% preços estão entre 40 e 80, mesmo sem saber a forma da distribuição de probabilidade dessa V.A.

---

## 🧮 Formulação Matemática de preços
O preço unitário $P_u$ é dado por:
$$
P_u = CMV \cdot (1 + M)
$$
Onde:
- $CMV$: Custo da Mercadoria Vendida se da pela soma dos custos de proteína, açúcar, cacau e embalagem.
- $M$: Margem bruta desejada.

O $CMV$ detalhado é:
$$
CMV_i = C_{\text{proteína}, i} + C_{\text{açúcar}, i} + C_{\text{cacau}, i} + C_{\text{embalagem}, i}
$$

### Intuição Matemática
Em teoria das probabilidades, o **valor esperado** de uma variável aleatória $X$ é:
$$
E[X] = \int x \cdot f(x)\,dx
$$
Na prática, usamos Monte Carlo para aproximar esse valor com uma média aritmética de $N$ amostras independentes:
$$
\hat{E}[P_u] = \frac{1}{N} \sum_{i=1}^{N} P_{u,i}
$$

Ou seja simulei o preço N vezes e tirei a média. 
            
Se $N=5$, isso seria explicitamente:
$$
\hat{E}[P_u] = \frac{1}{5}(P_{u,1} + P_{u,2} + P_{u,3} + P_{u,4} + P_{u,5})
$$

onde cada $P_{u,i}$ é o resultado da i-ésima simulação. Ou seja a simulação de Monte Carlo nos ajuda a estimar o  preço mais provável da barra de proteína, mesmo quando os custos variam muito. Em vez de depender de um único cenário fixo, usamos milhares de cenários possíveis e fazemos a média deles.

1. **Definição de espaço amostral $(\Omega, \mathcal{F}, P)$**:
   - $\Omega$: conjunto de todos os cenários possíveis (custos futuros dos insumos).
   - $\mathcal{F}$: conjunto de eventos (subconjuntos de $\Omega$). Uma V.A  é simplesmente um número que muda de forma imprevisível dentro de um certo padrão.
   - $P$: função probabilidade que atribui chance a cada evento.

2. **Variáveis aleatórias**:
   Cada custo (proteína, cacau, açúcar, embalagem) é modelado como uma função aleatória:
   $$
   X: \Omega \to \mathbb{R}
   $$
   onde $X(\omega)$ é o custo em um cenário específico $\omega$.
   Cada cenário possível (Ω) leva a um valor numérico (ℝ)

3. **Valor esperado**:
   O valor esperado é a média ponderada de todos os possíveis resultados:
   $$
   E[X] = \int_{\Omega} X(\omega)\,dP(\omega)
   $$

---

## 📝 Como Saber se a Estimativa é Boa?
✔️ **A qualidade da estimativa depende de:**
- **Número de Simulações ($N$):** quanto maior, mais estável a média (graças à LLN).
- **Desvio Padrão:** baixo desvio indica menor incerteza.
- **Intervalo [P5, P95]:** intervalo estreito significa maior confiabilidade. Esse intervalo contém 90% dos preços simulados.
- **Visualização das Distribuições:** valida se as suposições sobre volatilidade são realistas.

""")

# --- Simulação de Preço da Barra de Proteína ---
st.markdown("---")
st.header("\U0001F52C Simulação de Preço da Barra de Proteína")

st.sidebar.header("\U0001F4C5 Parâmetros de Entrada")
proteina_media = st.sidebar.number_input("Média do preço da proteína (R$/kg)", value=40.0)
proteina_sigma = st.sidebar.number_input("Desvio padrão da proteína (R$/kg)", value=5.0)
proteina_peso = st.sidebar.number_input("Peso da proteína por barra (kg)", value=0.03)
acucar_media = st.sidebar.number_input("Média do preço do açúcar (R$/kg)", value=20.0)
acucar_sigma = st.sidebar.number_input("Desvio padrão do açúcar (R$/kg)", value=3.0)
acucar_peso = st.sidebar.number_input("Peso do açúcar por barra (kg)", value=0.02)
cacau_media = st.sidebar.number_input("Média do preço do cacau (R$/kg)", value=25.0)
cacau_sigma = st.sidebar.number_input("Desvio padrão do cacau (R$/kg)", value=4.0)
cacau_peso = st.sidebar.number_input("Peso do cacau por barra (kg)", value=0.015)
emb_min = st.sidebar.number_input("Custo mínimo da embalagem (R$/unidade)", value=0.5)
emb_mode = st.sidebar.number_input("Custo mais provável da embalagem (R$/unidade)", value=0.6)
emb_max = st.sidebar.number_input("Custo máximo da embalagem (R$/unidade)", value=0.8)
margem = st.sidebar.slider("Margem bruta desejada (%)", 0, 100, 30) / 100
n_simulacoes = st.sidebar.number_input("Número de Simulações", value=10000)

np.random.seed(42)
C_proteina = np.random.normal(proteina_media, proteina_sigma, n_simulacoes) * proteina_peso
C_acucar = np.random.normal(acucar_media, acucar_sigma, n_simulacoes) * acucar_peso
C_cacau = np.random.normal(cacau_media, cacau_sigma, n_simulacoes) * cacau_peso
C_embalagem = np.random.triangular(emb_min, emb_mode, emb_max, n_simulacoes)

CMV = C_proteina + C_acucar + C_cacau + C_embalagem
preco_unitario = CMV * (1 + margem)

# Convergência da média
convergencia = np.cumsum(preco_unitario) / (np.arange(n_simulacoes) + 1)
fig_conv, ax_conv = plt.subplots()
ax_conv.plot(convergencia, label='Média acumulada')
ax_conv.axhline(np.mean(preco_unitario), color='red', linestyle='--', label='Média final')
ax_conv.set_title('📈 Convergência da Média Estimada')
ax_conv.set_xlabel('Número de Simulações')
ax_conv.set_ylabel('Preço Estimado (R$)')
ax_conv.legend()
st.pyplot(fig_conv)

# Pesos relativos
componentes = ['Proteína', 'Açúcar', 'Cacau', 'Embalagem']
medias = [np.mean(C_proteina), np.mean(C_acucar), np.mean(C_cacau), np.mean(C_embalagem)]
pesos = np.array(medias) / np.mean(CMV)

fig_pesos, ax_pesos = plt.subplots()
ax_pesos.pie(pesos, labels=componentes, autopct='%1.1f%%', startangle=90)
ax_pesos.set_title('📊 Participação no Custo Total (CMV)')
st.pyplot(fig_pesos)

# Estatísticas
media_preco = np.mean(preco_unitario)
desvio_preco = np.std(preco_unitario)
p5_preco = np.percentile(preco_unitario, 5)
p95_preco = np.percentile(preco_unitario, 95)

st.subheader("📊 Estatísticas da Simulação")
st.write(f"**Preço Médio Estimado:** R$ {media_preco:.2f}")
st.write(f"**Desvio Padrão:** R$ {desvio_preco:.2f}")
st.write(f"**5º Percentil (P5):** R$ {p5_preco:.2f}")
st.write(f"**95º Percentil (P95):** R$ {p95_preco:.2f}")

fig_hist, ax_hist = plt.subplots()
ax_hist.hist(preco_unitario, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax_hist.axvline(media_preco, color='red', linestyle='--', label='Média')
ax_hist.axvline(p5_preco, color='green', linestyle='--', label='P5')
ax_hist.axvline(p95_preco, color='orange', linestyle='--', label='P95')
ax_hist.set_title('Distribuição do Preço Unitário Simulado')
ax_hist.set_xlabel('Preço Unitário (R$)')
ax_hist.set_ylabel('Frequência')
ax_hist.legend()
st.pyplot(fig_hist)

# ACF e Lag Plot
st.subheader("📈 ACF, Lag Plot e Autocorrelação")
fig_lag, ax_lag = plt.subplots()
lag_plot(pd.Series(preco_unitario), ax=ax_lag)
ax_lag.set_title("Lag Plot - Preço Simulado")
st.pyplot(fig_lag)

fig_acf_sim, ax_acf_sim = plt.subplots()
plot_acf(preco_unitario, lags=30, ax=ax_acf_sim)
ax_acf_sim.set_title("ACF - Preço Simulado")
st.pyplot(fig_acf_sim)

acf_1 = acf(preco_unitario, nlags=1)[1]
st.markdown(f"### Autocorrelação (lag 1): `r = {acf_1:.4f}`")

# Runs Test
st.subheader("🧪 Runs Test - Aleatoriedade dos Preços")
def runs_test(series):
    median = np.median(series)
    runs, n1, n2 = 0, 0, 0
    prev = None
    for val in series:
        curr = 'A' if val >= median else 'B'
        if curr != prev:
            runs += 1
        if curr == 'A':
            n1 += 1
        else:
            n2 += 1
        prev = curr
    expected = (2 * n1 * n2) / (n1 + n2) + 1
    std = np.sqrt((2 * n1 * n2 * (2*n1*n2 - n1 - n2)) / (((n1 + n2)**2) * (n1 + n2 - 1)))
    z = (runs - expected) / std if std > 0 else 0
    p = 2 * (1 - norm.cdf(abs(z)))
    return runs, expected, z, p

r, e, z, p = runs_test(preco_unitario)
st.markdown(f"""
- Runs observados: `{r}`
- Esperado: `{e:.2f}`
- Estatística z: `{z:.2f}`
- **p-valor**: `{p:.4f}`

#### Hipóteses:
- **H₀:** Os preços simulados são aleatórios.
- **H₁:** Os preços simulados **não** são aleatórios.

✅ Como o **p-valor é alto**, não rejeitamos H₀.  
Os preços simulados mostram aleatoriedade estatisticamente aceitável.
""")
