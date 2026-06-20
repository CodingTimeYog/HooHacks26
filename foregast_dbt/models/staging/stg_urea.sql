{{ config(materialized='view') }}

with src as (
    select
        date_trunc('month', period)::date as month,
        urea_usd_per_mt
    from {{ source('raw', 'worldbank_pinksheet_monthly') }}
    where period is not null
      and urea_usd_per_mt is not null
),

deduped as (
    select
        month,
        avg(urea_usd_per_mt) as price_usd_per_mt
    from src
    group by month
)

select
    month,
    price_usd_per_mt
from deduped
