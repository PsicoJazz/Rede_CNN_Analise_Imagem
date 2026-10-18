# 🐾 Gato ou Cachorro? — Classificação de imagens com CNN

Este projeto nasceu como um estudo de redes neurais para reconhecimento de imagens. Eu comecei com o básico, treinando redes simples para reconhecer dígitos escritos à mão, e fui evoluindo até chegar em um modelo capaz de dizer se uma foto mostra um **gato** ou um **cachorro**.

No final, montei uma interface em **Streamlit** para que qualquer pessoa possa enviar uma foto e ver a resposta da rede na hora, sem precisar abrir o notebook.

## O que tem neste repositório

| Arquivo | Para que serve |
|---|---|
| `Redes_Neurais_CNN_Reconhecimento_de_imagens.ipynb` | O notebook com todo o estudo: dados, modelos, treino e avaliação |
| `Redes_Neurais_CNN_Reconhecimento_de_imagens.keras` | O modelo treinado de gatos e cachorros, pronto para uso |
| `app.py` | A interface em Streamlit |
| `requirements.txt` | As bibliotecas necessárias para rodar o app |

## Como o estudo foi feito

O notebook está dividido em duas partes.

### Parte 1: aquecimento com dígitos (MNIST)

Usei o dataset MNIST (imagens de 28×28 de números de 0 a 9) para comparar três tipos de rede e entender como a arquitetura influencia o resultado:

| Rede | Descrição | Acurácia no teste |
|---|---|---|
| Camada única (softmax) | Só uma camada densa, sem camadas ocultas | 92,9% |
| MLP | Uma camada oculta com 100 neurônios | 97,6% |
| CNN | Duas camadas de convolução com max pooling | 98,2% |

A CNN foi a melhor das três, e com bem poucos filtros (5 e 10). É um bom exemplo de como convoluções aproveitam a estrutura espacial da imagem.

### Parte 2: gatos e cachorros com transfer learning

Para o problema principal usei o **InceptionV3** pré-treinado no ImageNet. Em vez de treinar uma rede do zero, que exigiria muito mais dados e tempo, reaproveitei o que ela já aprendeu sobre bordas, texturas e formas.

O que foi feito:

- As camadas do InceptionV3 foram **congeladas** e o aproveitamento vai até a camada `mixed7`.
- Em cima dela adicionei uma camada `Flatten`, uma camada densa com 512 neurônios (ReLU), um `Dropout` de 20% e um neurônio final com ativação sigmoide, que devolve a probabilidade de a imagem ser um cachorro.
- As imagens de entrada têm tamanho **150×150** com 3 canais de cor, normalizadas dividindo por 255.
- O treino usou **data augmentation** (rotação, deslocamento, cisalhamento, zoom e espelhamento horizontal) para o modelo generalizar melhor com poucas imagens.
- O dataset é o `cats_and_dogs_filtered`, com 2.000 imagens de treino e 1.000 de validação, divididas igualmente entre as duas classes.
- Foram 10 épocas, com 100 passos por época.

**Resultado:** a acurácia de validação chegou a cerca de **97%** no melhor momento (época 7) e terminou em **94,6%** na última época. A oscilação no fim indica que o modelo começa a variar um pouco depois de certo ponto, então vale experimentar menos épocas ou um *early stopping* em versões futuras.

## Rodando a interface

### 1. Clonar o projeto

```bash
git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
cd SEU_REPOSITORIO
```

### 2. Instalar as dependências

Recomendo usar um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate      # no Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Garantir que o modelo está na pasta

O arquivo `.keras` precisa estar na mesma pasta do `app.py`. Veja a seção sobre o tamanho do arquivo mais abaixo.

### 4. Abrir o app

```bash
streamlit run app.py
```

O navegador abre sozinho em `http://localhost:8501`.

## O que a interface faz

- Aceita uma ou várias imagens de uma vez (JPG, PNG, WEBP e BMP).
- Também permite tirar uma foto direto pela câmera.
- Mostra a classe prevista, a confiança e a probabilidade de cada classe.
- Tem um controle de **confiança mínima** na barra lateral: se a rede não estiver segura o suficiente, o resultado aparece marcado como incerto.

## Limitações que vale conhecer

- **O modelo só conhece gatos e cachorros.** Se você enviar uma foto de um carro, de uma pessoa ou de uma paisagem, ele vai tentar encaixar em uma das duas classes, e às vezes com bastante confiança. Isso é esperado em um classificador binário.
- O dataset de treino é pequeno. Raças pouco comuns, fotos escuras ou animais muito pequenos na imagem podem gerar erros.
- Fotos com mais de um animal ou com os dois juntos também podem confundir a rede.

## Sobre o tamanho do arquivo do modelo

O arquivo `.keras` tem cerca de **268 MB**, e o GitHub não aceita arquivos acima de **100 MB** em um commit normal. Existem três caminhos:

1. **Git LFS**, que é o mais direto para quem quer manter tudo no repositório:
   ```bash
   git lfs install
   git lfs track "*.keras"
   git add .gitattributes
   ```
   Lembre que o plano gratuito do Git LFS tem limite de armazenamento e de banda.
2. **Releases do GitHub**, anexando o modelo como arquivo da release (aceita até 2 GB por arquivo) e orientando quem clonar a baixar de lá.
3. **Hugging Face Hub** ou Google Drive, deixando só o link no README.

## Tecnologias

- Python
- TensorFlow / Keras
- InceptionV3 (transfer learning)
- Streamlit
- NumPy e Pillow

## Possíveis próximos passos

- Fazer *fine-tuning* de algumas camadas finais do InceptionV3 para tentar melhorar a acurácia.
- Adicionar *early stopping* para evitar a queda de desempenho no fim do treino.
- Acrescentar uma terceira classe, como "nenhum dos dois", para lidar com imagens fora do esperado.
- Publicar o app online, por exemplo no Streamlit Community Cloud ou no Hugging Face Spaces.

## Autor

Feito por **SEU NOME**. Fique à vontade para abrir uma issue ou mandar sugestões.