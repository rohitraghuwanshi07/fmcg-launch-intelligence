# Case study: FMCG Launch Intelligence

**Question.** Should we launch this product, at this price, in these outlets - and what would make us regret it?

**Why it is hard.** A new product has no sales history; point forecasts hide the risk; assumptions are rarely labelled.

**What I built.** A leakage-safe cold-start model, a calibrated Monte-Carlo risk engine, and decision layers on top: readiness score, CVaR strategy optimiser, Red Team reverse stress test, Sobol sensitivity, conformal intervals, analog-launch cross-check, Thompson-sampling pilot planner, a grounded AI analyst, and a FastAPI service.

**Real evidence used.** Conditional-logit price/promotion response on six real supermarket scanner panels (own-price elasticities -1.7 to -3.4 with bootstrap CIs) and a global direct multi-horizon LightGBM on 152 real retail series that beats seasonal-naive at every horizon (WAPE 2.2% to 3.7% vs 4.4%) with calibrated conformal bands.

**What I would not claim.** The launch curve is validated on simulated data only; marketing, channel and regional-surprise parameters are assumptions; scanner elasticities are brand-switching estimates from historical US panels.

**Next step.** One real launch panel (retailer scan data with introduction dates) to validate the cold-start curve; then close the loop with the pilot planner.
