"""Load real sensor classes with deterministic HA infrastructure adapters."""
import ast
import asyncio
from collections import Counter, defaultdict, deque
from datetime import datetime, date, time, timezone, timedelta
import importlib.util
import json
import logging
import os
from pathlib import Path
import time as time_module
from types import SimpleNamespace
import urllib.parse
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'custom_components/pjm'


def module(name):
    spec = importlib.util.spec_from_file_location('pjm_test_'+name, SOURCE/(name+'.py'))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class Clock:
    DEFAULT_TIME_ZONE = ZoneInfo('America/Chicago')
    value = datetime(2026,7,1,15,tzinfo=DEFAULT_TIME_ZONE)
    @classmethod
    def now(cls): return cls.value
    @classmethod
    def utcnow(cls): return cls.value.astimezone(timezone.utc)
    @classmethod
    def as_local(cls, value): return value.astimezone(cls.DEFAULT_TIME_ZONE)
    @staticmethod
    def as_utc(value): return value.astimezone(timezone.utc)
    @staticmethod
    def parse_datetime(value): return datetime.fromisoformat(value)


class Store:
    def __init__(self,*args): self.data=None
    async def async_load(self): return self.data
    async def async_save(self,data): self.data=data.copy()


class Entity:
    async def async_added_to_hass(self): pass
    async def async_will_remove_from_hass(self): pass
    def async_write_ha_state(self): pass


class Hass:
    config = SimpleNamespace(path=lambda x:x)
    @staticmethod
    def async_create_task(coro): coro.close()


class ClientError(Exception): pass
class ClientTimeout:
    def __init__(self,total): self.total=total


def finish(coro):
    try: coro.send(None)
    except StopIteration as done: return done.value
    else:
        coro.close()
        raise AssertionError('Unexpected real asynchronous I/O')


def load_sensor():
    timers=[]
    def track(hass,action,stamp):
        record=dict(action=action,stamp=stamp,cancelled=False)
        timers.append(record)
        def cancel(): record['cancelled']=True
        return cancel
    ns=dict(asyncio=asyncio,Counter=Counter,defaultdict=defaultdict,deque=deque,
            datetime=datetime,date=date,time=time,timezone=timezone,timedelta=timedelta,
            json=json,logging=logging,os=os,urllib=urllib,time_module=time_module,
            np=np,SensorEntity=Entity,SensorStateClass=SimpleNamespace(MEASUREMENT='measurement'),
            Store=Store,dt_util=Clock,_LOGGER=logging.getLogger('release_tests'),
            Throttle=lambda _:lambda fn:fn,async_track_point_in_utc_time=track,
            aiohttp=SimpleNamespace(ClientError=ClientError,ClientTimeout=ClientTimeout),
            CurtailmentBudget=module('curtailment').CurtailmentBudget,
            parse_prices=module('prices').parse_prices)
    exec(compile(ast.parse((SOURCE/'const.py').read_text()),'const.py','exec'),ns)
    tree=ast.parse((SOURCE/'sensor.py').read_text(encoding='utf-8'))
    assignments=[n for n in tree.body if isinstance(n,ast.Assign)
                 and all(isinstance(t,ast.Name) for t in n.targets)
                 and not any(t.id=='_LOGGER' for t in n.targets)]
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
           and n.name in ('_quadratic','_format_peak_history_entry','PJMSensor',
                          'CoincidentPeakPredictionSensor','PJMData','parse_da_hrl_lmp_items')]
    exec(compile(ast.Module(body=assignments+nodes,type_ignores=[]),'sensor.py','exec'),ns)
    return SimpleNamespace(**ns),timers
