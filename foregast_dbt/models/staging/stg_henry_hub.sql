{{ config(materialized='view') }}

-- Henry Hub spot price, monthly.
-- Union the EIA daily series (aggregated to monthly mean — matches the
-- live.resample("MS").mean() step in backend/src/ingestion/pipeline.py)
-- with the local monthly CSV backbone. EIA wins on overlap and extends
-- beyond the local file's last month.

with eia_monthly as (
    select
        date_trunc('month', period)::date as month,
        avg(value_usd_per_mmbtu)          as price_usd_per_mmbtu,
        1                                 as source_priority   -- lower = preferred
    from {{ source('raw', 'eia_ng_spot_daily') }}
    where period is not null
      and value_usd_per_mmbtu is not null
    group by 1
),

local_monthly as (
    select
        date_trunc('month', period)::date as month,
        price_usd_per_mmbtu,
        2                                 as source_priority
    from {{ source('raw', 'henry_hub_spot_monthly_csv') }}
    where period is not null
      and price_usd_per_mmbtu is not null
),

combined as (
    select * from eia_monthly
    union all
    select * from local_monthly
)

select distinct on (month)
    month,
    price_usd_per_mmbtu
from combined
order by month, source_priority
