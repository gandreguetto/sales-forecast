# Aplicando Machine Learning na análise de séries temporais para prever as vendas da loja equatoriana "Favorita"

## Uma visão geral sobre o problema e as técnicas utilizadas

Nesta análise são utilizadas técnicas de Machine Learning na análise de séries temporais de vendas de produtos da rede de lojas equatoriana "Favorita".

Os dados foram obtidos do Kaggle (https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) e fornecem informações sobre as vendas de 33 famílias de produtos em 54 lojas da rede. Na tabela principal tem-se a quantidade de itens de cada família vendida por dia em cada loja e também a quantidade de itens em promoção. Outras variáves externas, que potencialmente influenciam as vendas, são sugeridas. São elas:

- **Valor do petróleo.** A economia do Equador é bastante dependente do petróleo e, portanto, o seu valor pode influenciar o comportamento de compras da população.
- **Feriados.**
- **Dias de pagamento dos funcionários públicos.** No Equador, os funcionários públicos são pagos quinzenalmente, no dia 15 e no último dia do mês. 

Nessa primeira análise, apenas uma loja e uma família de produtos foram sorteados para as análises. Para aplicar técnicas de Machine Learning na previsão de vendas da série temporal a função de autocorrelação parcial foi utilizada. A função de autocorrelação parcial é derivada da função de autocorrelação e mede a correlação das vendas com os dias anteriores mas penalizando padrões que se repetem. Por exemplo, se algum valor tem forte correlação com o valor da semana anterior, como é o caso das vendas que será análisado, a função de autocorrelação terá um comportamento períodico com picos a cada semana de atraso na função. A função de autocorrelação parcial atenua esses picos conforme eles se repetem.

Uma vez calculadas as autocorrelações parciais, os atrasos ("lags") que geram os sinais de maior intensidade são utilizados como "features" na modelagem. O problema de análise de uma série temporal é, dessa forma, transformado em um problema tradicional de Machine Learning, no qual as correlações temporais dos dados são transformadas em features, que são os valores das vendas nos devidos "lags" principais. 

Uma das principais vantagens dessa técnica é que váriaveis podem ser incluídas na análise de uma série temporal de forma relativamente simples. Outra vantagem importante é a possibilidade de se utilizar os diversos algorítmos de Machine Learning existentes.  

Aqui, o algorítmo XGBoost será utilizado no problema de regressão para se prever as vendas em uma janela de 15 dias. Conforme será visto, as melhores previsões são obtidas quando as "features" mencionadas acima são utilizadas excluindo-se os dias de pagamento.

## A série de vendas

Antes do pré-processamento a série temporal de vendas da família "HOME CARE" da loja 41 tem o comportamento ilustrado na figura abaixo.

![time_series_1](https://user-images.githubusercontent.com/88217999/214460691-c7875ebb-5679-4ed5-8c81-31a9a4c78eed.png)

Após a eliminação da parte inicial, onde existem longos períodos sem vendas, e do outlier com venda muito acima da média, a série se comporta conforme ilustrado na figura abaixo.

![time_series_2](https://user-images.githubusercontent.com/88217999/214461472-b2f89157-828b-46bb-a885-2f2ba72cd308.png)

## As funções de autocorrelação e autocorrelação parcial

A função de autocorrelação (ou, do inglês, acf) calcula a correlação dos dados com os próprios dados da série em períodos anteriores. 

Na imagem abaixo, o pacote statsmodels é utilizado para vizualizar a acf da série. Nesse caso, foi escolhida a opção de mostrar as correlações para um período de até 40 dias. 

Na imagem, observa-se a correlação trivial com o período anterior ("lag") de zero dias, que é obviamente igual a 1. Ou seja, o coeficiente de correlação das vendas de um dia com esse mesmo dia é igual a 1, como esperado.   

Existem correlações positivas com "lags" de 1, 6, 7, 8, 13, 14, 15, 20, 21, 22 ... dias. E correlações negativas ocorrem nos "lags" de 2, 3, 4, 5, 9, 10, 11, 12, ... dias. Nota-se uma forte correlação com as vendas de 7 dias atrás, indicando uma forte dependência nas vendas com o dia da semana. Essa dependência com o dia da semana ainda é responsável pelo padrão periódico observado. 

Para selecionar somente os "lags" principais dentro desse padrão periódico é utilizada a função de autocorrelação parcial (que muitas vezes é escrita pela sigla em inglês pacf). Mais abaixo o pacote statsmodels é novamente utilizado para produzir o gráfico da pacf da série. Na figura fica bastante claro a atenuação dos valores da correlação conforme avança-se no padrão repetitivo. 

![acf](https://user-images.githubusercontent.com/88217999/214462822-f907cc15-943d-4510-a267-94a90b12751f.png)

![pacf](https://user-images.githubusercontent.com/88217999/214462508-efe94120-77a6-4bac-9a62-37349f870b48.png)

Os "lags" principais, que serão utilizados no modelo de Machine Learning, são selecionados como aqueles com valor absoluto de autocorrelação maior do que 0.2. Na figura acima a linha cinza horizontal indica esse limiar.  

## Construindo o dataframe para a aplicação das técnicas de Machine Learning

O ponto central da técnica de Machine Learning para a análise de Séries Temporais que é utilizada aqui consiste em transformar os atrasos ("lags") principais em "features" para o aprendizado. 

Portanto, inicialmente esses atrasos principais são identificados. Têm-se 5 valores de atraso que serão utilizados nos modelos. Ou seja, as vendas em um determinado dia têm como variáveis os valores de venda a 1, 2, 6, 7, e 14 dias atrás.

Além dessas features, de forma geral também são considerados na análise o número de itens em promoção, o preço do petróleo, os feriados e as datas de pagamento do setor público.
