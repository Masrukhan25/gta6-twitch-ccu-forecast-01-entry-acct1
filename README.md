# GTA VI Twitch CCU Forecast

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/competitions/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Masrukhan25/gta6-twitch-ccu-forecast-01-entry-acct1/rank.svg)](https://make.fun/competitions/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Masrukhan25/gta6-twitch-ccu-forecast-01-entry-acct1/reward.svg)](https://make.fun/competitions/gta6-twitch-ccu-forecast-01)

Forecasts Grand Theft Auto VI's launch-day Twitch viewers hour by hour (hours 2–24) as five quantiles each, from the hour-1 print and the release-day shapes of 38 past big game launches.

## Method

For each hour, the forecast is `p1 * exp(weighted quantile of l_h)`, where `p1` is the hour-1 print and `l_h = ln(viewers_h / viewers_1)` is each comparable's release-day log-ratio. A comparable's weight is a similarity kernel over platform match (console-only vs multi-platform), recency (3-year half-life), log hour-1 viewers distance, pre-release buzz (mean hourly viewers in the 24 h before release, relative to hour 1), and research tier. The target's own category history never enters the shape pool. No per-hour bias correction and no spread scaling — leave-one-out CV over the comparables showed both raise the loss.

## Validation (leave-one-out over the 37 usable comparables)

| | Mean pinball loss |
|---|---|
| This model, all comparables | 0.413 |
| This model, relevant subset | 0.278 |
| This model, practice-like targets | 0.169 |
| House persistence baseline, all | 0.930 |
| House persistence baseline, relevant | 0.360 |
| House persistence baseline, practice-like | 0.301 |

## License

MIT — see [LICENSE](LICENSE).
