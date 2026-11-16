<!-- SI-assisted: written with Claude (Anthropic), reviewed by the Space Monkeys team. See README. -->

# Slide script: Harmonizing MODIS and VIIRS hotspots

Seven slides. "On screen" is what the slide shows; "Talk" is what the presenter says. Every number comes from [methodology.md](methodology.md) and the [README](../README.md).

## Slide 1 — The record breaks in 2012

**On screen**
- Figure: [brazil_annual.png](figures/brazil_annual.png). The naive series (MODIS until 2012, VIIRS after) jumps in 2012.
- "Counting VIIRS as if it were MODIS overstates fire by +145% in Brazil."
- "This step is the sensor, not the fire. It would dominate any trend."
- "MODIS ends in 2027. 2012–2021 is the only window to learn the bridge."

**Talk:** A hotspot is a hot pixel, not a fire. VIIRS sees smaller fires and splits one fire into more pixels. Without harmonization, "is burning going up?" gets the wrong answer.

## Slide 2 — Framing decisions

**On screen**
- Unit: the fire cell-day (0.1° cell × UTC day). Sensors are combined with OR, never a sum.
- MODIS is the unit of the record: only it covers 20+ years. VIIRS is expressed in MODIS-equivalent units.
- Reference years 2012–2021 only: Terra and Aqua orbits drift after that (Terra 10:26 → 8:50, Aqua 13:40 → 15:30 local time).
- One model per VIIRS satellite: "VIIRS combined" grows as NOAA-20 and NOAA-21 join, while P(MODIS | S-NPP) stays stable (0.60 / 0.59 / 0.57).

**Talk:** Every decision here came from the data. We found the orbit drift because two different models drifted the same way.

## Slide 3 — Algorithm 1: "Would MODIS have seen this?"

**On screen**
- The question: P(MODIS detected fire in this cell-day | what one VIIRS satellite saw).
- Features: pixels, FRP, day/night, 3×3 and 5×5 neighbourhood, lags up to 3 days, seasonality. No feature uses MODIS, because MODIS will not exist when the model is used.
- Temporal split (never random), weighted positive class, isotonic calibration.

| Model (Pantanal, test 2018–2026) | Bias | Monthly WAPE |
|---|---|---|
| Naive | +103.8% | 103.8% |
| Ratio | −13.0% | 20.3% |
| **LightGBM** | **+3.7%** | **5.6%** |

- Brazil: LightGBM 6.9% vs U-Net 6.5% monthly WAPE.

**Talk:** The U-Net was our bet, and it won, but by 0.4 points. That is smaller than the uncertainty we already carry, and putting it in the record would cost about 5 GPU hours of cross-fitting. We serve LightGBM and keep the U-Net comparison in the repository as evidence. About 70% of the gain comes from FRP: MODIS misses weak fires.

## Slide 4 — Algorithm 2: from probabilities to a record

**On screen**
1. Cross-fitting over blocks of years: every reference estimate is out of sample.
2. Isotonic calibration per biome: one national map left the Pantanal 6% high and Caatinga and Pampa 11–13% low.
3. Coverage gap g per biome and month: corrects for MODIS fires no VIIRS-based model can see (small fires seen by Terra at about 10:30).
4. 90% interval from the monthly errors of the reference years.

- Result: monthly WAPE 3.7% for Brazil, 4.9–14.1% per biome, against +145% with no correction.

**Talk:** The output is not a yes or no. It is an expected count of MODIS fire cell-days, with its uncertainty stated.

## Slide 5 — Algorithm 3: carrying the record past MODIS

**On screen**
- Application table: the same features, built from VIIRS alone, already run to 2026-11-14 (NRT flagged provisional).
- Bridge chain: NOAA-21 never flew with stable MODIS, so it learns from NOAA-20. Each link is one model.
- Control: NOAA-20 through S-NPP scores 4.1% monthly WAPE vs 6.3% directly. The extra link adds almost no error.
- VIIRS-only layer: each detection = p (MODIS would see it too) + (1 − p) (only VIIRS sees it). Nothing is lost; the layers add back to what was observed.

**Talk:** This is what keeps the method alive after 2027. It can pass through any number of satellite changes, one link at a time.

## Slide 6 — Validation, honestly

**On screen**
- Figure: [pantanal_2020.png](figures/pantanal_2020.png). Within ±25% of MODIS every month, +11% to +13% at the peak. S-NPP as seen is 29% to 112% above.
- The 2012 step is gone: yearly error from −3.5% to +3.1%, 2012–2021.
- Second country, no code change: Australia, 2.6% monthly WAPE, same error pattern by region.
- Honest limit: after 2021 observed MODIS moves away from the estimate (−15% in 2025). The timing matches the orbit drift, but we have not separated drift from model error.

**Talk:** We show where the method stops, not only where it works.

## Slide 7 — What responders get

**On screen**
- A calendar for any area (box, biome or drawn polygon):
  - normal year (median and P10–P90);
  - anomaly (percentile; record / above / normal / below);
  - season (start at 10% of the year's fire, peak, end at 90%);
  - trend (Theil–Sen with a 95% interval, robust to one extreme year).
- The API never runs a model: everything is read from a Parquet cube with DuckDB.
- Bonus: 7-day fire forecast (U-Net, average precision 0.396 vs 0.303 for climatology).
- Links / QR codes: website · repository · video.
