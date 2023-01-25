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
