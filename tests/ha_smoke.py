"""Run with real HA installed; deliberately separate from offline adapters."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import custom_components.pjm.sensor
from custom_components.pjm.config_flow import PJMConfigFlow, PJMOptionsFlowHandler


async def main():
    flow=PJMConfigFlow()
    schema=flow._build_schema()
    values=schema({'zone':'COMED'})
    assert values['cover_mode']=='secure'
    assert values['max_daily_curtailment_minutes']==120
    options=PJMOptionsFlowHandler(SimpleNamespace(data={}))
    selected=options._build_schema(dict(zone='COMED',instantaneous_total_load=True,
        cover_mode='intermediate',max_daily_curtailment_minutes=120))
    values=selected({'zone':'COMED'})
    assert values['instantaneous_total_load'] is True
    assert values['cover_mode']=='intermediate'
    print('PASS: real HA imports and both form schemas')


asyncio.run(main())
