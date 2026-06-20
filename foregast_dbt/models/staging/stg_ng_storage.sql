{{ config(materialized='view') }}

-- US Lower-48 working gas storage, monthly (MMcf).
-- Union the EIA weekly series (last observation within each calendar month,
-- BCF -> MMcf via ×1000 — matches live.resample("MS").last() and *1000 in
-- backend/src/ingestion/pipeline.py) with the local monthly XLS backbone.
-- EIA wins on overlap and extends beyond the local file's last month.

with eia_weekly_last_in_month as (
    select distinct on (date_trunc('month', period))
        date_trunc('month', period)::date as month,
        value_bcf * 1000.0                as storage_mmcf,
        1                                 as source_priority
    from {{ source('raw', 'eia_ng_storage_weekly') }}
    where period is not null
      and value_bcf is not null
    order by date_trunc('month', period), period desc
),

local_monthly as (
    select
        date_trunc('month', period)::date as month,
        working_gas_mmcf                  as storage_mmcf,
        2                                 as source_priority
    from {{ source('raw', 'eia_ng_storage_monthly_xls') }}
    where period is not null
      and working_gas_mmcf is not null
),

combined as (
    select * from eia_weekly_last_in_month
    union all
    select * from local_monthly
)

select distinct on (month)
    month,
    storage_mmcf
from combined
order by month, source_priority
