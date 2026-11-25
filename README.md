# Linguagem de desinformação em português: o que os ~98% realmente medem?

Projeto da disciplina de Deep Learning. Tem duas partes:

1. **Pesquisa.** A literatura reporta ~98% de acurácia na detecção de *fake news* em português. Investigamos se esse número mede a detecção de desinformação ou o **estilo/fonte** dos textos do dataset.
2. **Ferramenta.** Um app de apoio à leitura crítica que combina três modelos (BERTimbau, e5 e Qwen3) e uma interface Gradio.

> ⚠️ **O app nunca diz que uma notícia é falsa ou verdadeira.** Ele mede o quanto o texto *se parece* com a linguagem dos textos de desinformação do treino e mostra checagens reais parecidas, quando existem.

## Conclusão em uma frase

Em split aleatório do FakeRecogna 2.0, um modelo que **nem lê o texto** (só a URL) acerta 99,2%, mais que o BERTimbau (98,4%). Fora do dataset, o BERT cai para 64,6%. **Os ~98% medem a distinção entre gêneros textuais (artigo de checagem × reportagem), não a detecção de desinformação.**

## Principais resultados

Acurácia no teste do FakeRecogna (7.920 textos), salvo quando indicado.

| Experimento | Resultado |
|---|---|
| **Teste de atalho** (só metadado, o texto nunca entra) | URL (domínio + 1ª pasta) **99,19%** · categoria 97,35% · autor 95,83% · formato da data 94,48% · domínio 90,96% |
| TF-IDF + Regressão Logística | 95,42% (cortado em 50 palavras: 93,89%) |
| **BERTimbau** | **98,37%** (F1 98,37; AUC 0,998; IC95% 98,08–98,66) |
| BERTimbau no **Fake.Br** (fora do dataset) | **64,56%** (F1 64,94; AUC 0,715; IC95% 63,42–65,76) |
| McNemar BERT × atalho da URL | só o BERT acerta 51, só o atalho acerta 116, p ≈ 7e-7 → **o atalho supera o BERT** |
| TF-IDF cruzado | FakeRecogna→Fake.Br 72,88% · Fake.Br→FakeRecogna 51,87% (dentro de cada um: 95,42% / 97,29%) |
| Superconfiança | no Fake.Br, com confiança ≥ 0,99 o BERT acerta só 71,3% |
| BERT por fonte | 97,55% em g1/UOL (únicas fontes com as duas classes) × 98,7% nas demais |
| Máscara do vocabulário de checagem (TF-IDF) | 95,42% → 94,91%: quase sem efeito; o sinal de gênero está espalhado no texto |
| Quase-duplicatas treino/teste | 3,2% do teste tem cosseno > 0,8 com algum texto de treino; o BERT acerta **menos** nelas (96,8% × 98,4%): o vazamento não explica o resultado |
| RAG (limiar 0,90) | 30 paráfrases: top-1 86,7%, top-3 100%; 0/15 falsos casamentos · 100 notícias reais: **0/100** acima do limiar (IC95% 0–4%), mas o maior score foi 0,895, colado no limiar |

Palavras de maior peso no TF-IDF: **fake** → *circula, facebook, vídeo, publicações, verdade, uol confere, whatsapp, falsa, enganosa*; **real** → *disse, hoje, nesta, afirmou, segundo, ontem*. Os textos "fake" do dataset são, na prática, **artigos que desmentem boatos**.

## Como funciona o app

```
texto ──► BERTimbau ──► pontuação (0–1) + oclusão (palavras que mais pesaram)
   │
   ├────► e5 + FACTCK.BR ──► checagens com similaridade ≥ 0,90 (veredito e data do veículo)
   │
   └────► laudo JSON ──► Qwen3-1.7B ──► explicação em português ──► Gradio
```

Proteções: a conclusão é calculada em código (o LLM só explica e não recebe a pontuação); o texto do usuário é tratado como dado (defesa contra *prompt injection*); links fora do laudo são removidos; avisos fixos sobre os limites.

## Como rodar

Os notebooks rodam no Google Colab com GPU (T4) e usam a pasta `/content/drive/MyDrive/fakenews` no Drive.

1. **`notebooks/Pipeline_FakeNews_pd.ipynb`** baixa os dados, roda todos os experimentos, treina o BERTimbau (~20 min na T4) e salva os artefatos no Drive.
2. **`notebooks/Interface_Fakenews_app_pd.ipynb`** carrega só os artefatos e abre a interface (alguns minutos). Sem GPU, roda na CPU com o Qwen3-0.6B no lugar do 1.7B (só a explicação muda).

```
pip install -r requirements.txt
```

## Estrutura

```
├── README.md
├── apresentacao.pdf                        # slides da apresentação
├── requirements.txt
└── notebooks/
    ├── Pipeline_FakeNews_pd.ipynb          # experimentos: atalho, TF-IDF, BERTimbau, avaliação, investigação, RAG
    ├── Interface_Fakenews_app_pd.ipynb     # aplicativo: BERTimbau + RAG + Qwen3 + Gradio
    └── rag_fakenews.ipynb                  # exploração e calibração do limiar do RAG
```

Os artefatos (modelo ajustado `bertimbau_final/` de ~438 MB, previsões, índice do RAG e `resultados_atalho.json`) ficam numa pasta do Google Drive (`/MyDrive/fakenews`) e não são versionados. Os datasets são baixados pelos próprios notebooks.

## Limitações

- **Uma seed** de treino; o IC mede só a incerteza da amostra de teste.
- Split **aleatório** (não temporal); textos de 128 tokens (≈ 90 palavras), o resto é truncado.
- O rótulo do FakeRecogna vem da **fonte**: os "fakes" são artigos de checagem. A queda no Fake.Br mistura vários efeitos (definição de fake, tamanho, época).
- A máscara de vocabulário é parcial; os testes mostram **correlação**, não o que o BERT usa internamente.
- Viés de assunto: "lula" é um dos termos de maior peso para "fake".
- RAG: base pequena (~1,3 mil alegações de 3 agências, 2018–2019); embeddings tratam mal **negações**; as 30 paráfrases da calibração foram **rascunhadas por um LLM sem edição**, e o mesmo conjunto escolheu e reportou o limiar. Ausência de checagem não significa que o texto seja verdadeiro.
- "Heurísticas de persuasão" são regras não validadas.

## Trabalhos futuros

Split temporal; leave-one-source-out; remoção de quase-duplicatas; várias seeds; calibração (temperature scaling); NLI após a busca (para tratar negações); busca híbrida (BM25 + embeddings); dataset sem viés de fonte.

## Dados e modelos

| Recurso | Uso | Link |
|---|---|---|
| FakeRecogna 2.0 (extrativo) | treino/teste | https://huggingface.co/datasets/recogna-nlp/fakerecogna2-extrativo |
| Fake.Br Corpus | teste fora do dataset | https://github.com/roneysco/Fake.br-Corpus |
| FACTCK.BR | base do RAG | https://github.com/jghm-f/FACTCK.BR |
| BERTimbau | classificador | https://huggingface.co/neuralmind/bert-base-portuguese-cased |
| multilingual-e5-small | embeddings | https://huggingface.co/intfloat/multilingual-e5-small |
| Qwen3-1.7B | explicação | https://huggingface.co/Qwen/Qwen3-1.7B |

Respeite as licenças de cada recurso (veja o `LICENSE` ou README de cada repositório). As referências bibliográficas de cada dataset e modelo estão nos READMEs das respectivas páginas.
