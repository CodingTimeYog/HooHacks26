{{ config(materialized='table') }}

-- Monthly commodity panel — exact shape that run_ingestion() returns:
--   month, ng_spot, urea, dap, storage_mmcf
-- Built as a 4-way full outer join via a months CTE so a missing month in any
-- one staging model still yields a row (with NULLs in that column). engineer.py
-- will consume this directly once parity is confirmed.

with months as (
    select month from {{ ref('stg_henry_hub') }}
    union
    select month from {{ ref('stg_urea') }}
    union
    select month from {{ ref('stg_dap') }}
    union
    select month from {{ ref('stg_ng_storage') }}
)

select
    m.month,
    hh.price_usd_per_mmbtu as ng_spot,
    u.price_usd_per_mt     as urea,
    d.price_usd_per_mt     as dap,
    ng.storage_mmcf        as storage_mmcf
from months m
left join {{ ref('stg_henry_hub') }}  hh on hh.month = m.month
left join {{ ref('stg_urea') }}       u  on u.month  = m.month
left join {{ ref('stg_dap') }}        d  on d.month  = m.month
left join {{ ref('stg_ng_storage') }} ng on ng.month = m.month
order by m.month
