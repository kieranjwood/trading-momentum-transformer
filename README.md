# Trading with the Momentum Transformer
## About
This code accompanies the paper [Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture](https://arxiv.org/pdf/2112.08534.pdf) and additionally provides an implementation for the paper [Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection](https://arxiv.org/pdf/2105.13727.pdf). 

## Using the code
1. Create a Nasdaq Data Link account to access the [free Quandl dataset](https://data.nasdaq.com/data/CHRIS-wiki-continuous-futures/documentation). This dataset provides continuous contracts for 600+ futures, built on top of raw data from CME, ICE, LIFFE etc.
2. Download the Quandl data with: `python -m data.download_quandl_data <<API_KEY>>`
3. Create Momentum Transformer input features with: `python -m examples.create_features_quandl`. In this example we use the 100 futures tickers which have i) the longest history ii) more than 90% of trading days have data iii) data up until at least Dec 2021.
4. Optionally, run the changepoint detection module: `python -m examples.concurent_cpd_quandl <<CPD_WINDOW_LENGTH>>`, for example `python -m examples.concurent_cpd_quandl 21` and `python -m examples.concurent_cpd_quandl 126`
5. Create Momentum Transformer input features, including CPD module features with: `python -m examples.create_features_quandl 21` after the changepoint detection module has completed.
6. To create a features file with multiple changepoint detection lookback windows: `python -m examples.create_features_quandl 126 21` after the 126 day LBW changepoint detection module has completed and a features file for the 21 day LBW exists.
7. Run one of the Momentum Transformer or Slow Momentum with Fast Reversion experiments with `python -m examples.run_dmn_experiment <<EXPERIMENT_NAME>>`

## Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture
> Deep learning architectures, specifically Deep Momentum Networks (DMNs) , have been found to be an effective approach to momentum and mean-reversion trading. However, some of the key challenges in recent years involve learning long-term dependencies, degradation of performance when considering returns net of transaction costs and adapting to new market regimes, notably during the SARS-CoV-2 crisis. Attention mechanisms, or Transformer-based architectures, are a solution to such challenges because they allow the network to focus on significant time steps in the past and longer-term patterns. We introduce the Momentum Transformer, an attention-based architecture which outperforms the benchmarks, and is inherently interpretable, providing us with greater insights into our deep learning trading strategy. Our model is an extension to the LSTM-based DMN, which directly outputs position sizing by optimising the network on a risk-adjusted performance metric, such as Sharpe ratio. We find an attention-LSTM hybrid Decoder-Only Temporal Fusion Transformer (TFT) style architecture is the best performing model. In terms of interpretability, we observe remarkable structure in the attention patterns, with significant peaks of importance at momentum turning points. The time series is thus segmented into regimes and the model tends to focus on previous time-steps in alike regimes. We find changepoint detection (CPD) , another technique for responding to regime change, can complement multi-headed attention, especially when we run CPD at multiple timescales. Through the addition of an interpretable variable selection network, we observe how CPD helps our model to move away from trading predominantly on daily returns data. We note that the model can intelligently switch between, and blend, classical strategies - basing its decision on patterns in the data.

## Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection
> Momentum strategies are an important part of alternative investments and are at the heart of commodity trading advisors (CTAs). These strategies have, however, been found to have difficulties adjusting to rapid changes in market conditions, such as during the 2020 market crash. In particular, immediately after momentum turning points, where a trend reverses from an uptrend (downtrend) to a downtrend (uptrend), time-series momentum (TSMOM) strategies are prone to making bad bets. To improve the response to regime change, we introduce a novel approach, where we insert an online changepoint detection (CPD) module into a Deep Momentum Network (DMN) pipeline, which uses an LSTM deep-learning architecture to simultaneously learn both trend estimation and position sizing. Furthermore, our model is able to optimise the way in which it balances 1) a slow momentum strategy which exploits persisting trends, but does not overreact to localised price moves, and 2) a fast mean-reversion strategy regime by quickly flipping its position, then swapping it back again to exploit localised price moves. Our CPD module outputs a changepoint location and severity score, allowing our model to learn to respond to varying degrees of disequilibrium, or smaller and more localised changepoints, in a data driven manner. Back-testing our model over the period 1995-2020, the addition of the CPD module leads to an improvement in Sharpe ratio of one-third. The module is especially beneficial in periods of significant nonstationarity, and in particular, over the most recent years tested (2015-2020) the performance boost is approximately two-thirds. This is interesting as traditional momentum strategies have been underperforming in this period.


## References
Please cite our papers with:
```bib
@article{wood2021trading,
  title={Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture},
  author={Wood, Kieran and Giegerich, Sven and Roberts, Stephen and Zohren, Stefan},
  journal={arXiv preprint arXiv:2112.08534},
  year={2021}
}

@article {Wood111,
	author = {Wood, Kieran and Roberts, Stephen and Zohren, Stefan},
	title = {Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection},
	volume = {4},
	number = {1},
	pages = {111--129},
	year = {2022},
	doi = {10.3905/jfds.2021.1.081},
	publisher = {Institutional Investor Journals Umbrella},
	issn = {2640-3943},
	URL = {https://jfds.pm-research.com/content/4/1/111},
	eprint = {https://jfds.pm-research.com/content/4/1/111.full.pdf},
	journal = {The Journal of Financial Data Science}
}
```

The Momentum Transformer uses a number of components from the Temporal Fusion Transformer (TFT). The code for the TFT can be found [here](https://github.com/google-research/google-research/tree/master/tft).

## Sample results
Will be made available soon. 
