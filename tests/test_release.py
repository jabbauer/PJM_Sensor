import ast
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from support import Clock, Hass, SOURCE, ROOT, ClientError, finish, load_sensor, module


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.cls=module('curtailment').CurtailmentBudget
        self.now=datetime(2026,7,1,15,tzinfo=ZoneInfo('America/Chicago'))

    def test_interrupted_intervals_and_cap(self):
        b=self.cls()
        self.assertTrue(b.advance(self.now,True,7200))
        b.advance(self.now+timedelta(minutes=45),False,7200)
        b.advance(self.now+timedelta(minutes=60),True,7200)
        self.assertFalse(b.advance(self.now+timedelta(minutes=135),True,7200))
        self.assertEqual(b.seconds,7200)

    def test_restart_charges_active_downtime(self):
        b=self.cls(); b.advance(self.now,True,7200)
        restored=self.cls()
        restored.restore(b.snapshot(),self.now+timedelta(minutes=100),7200)
        self.assertEqual(restored.seconds,6000)
        self.assertFalse(restored.active)
        self.assertTrue(restored.advance(self.now+timedelta(minutes=100),True,7200))
        self.assertEqual(restored.deadline(self.now+timedelta(minutes=100),7200),
                         (self.now+timedelta(minutes=120)).astimezone(timezone.utc))

    def test_new_day_and_corrupt_usage(self):
        b=self.cls(); b.advance(self.now,True,60)
        b.advance(self.now+timedelta(days=1),False,60)
        self.assertEqual(b.seconds,0)
        with self.assertRaises(ValueError):
            b.restore(dict(day=self.now.date().isoformat(),seconds=float('nan')),self.now,60)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.ns,self.timers=load_sensor()
        Clock.value=datetime(2026,7,1,15,2,tzinfo=Clock.DEFAULT_TIME_ZONE)
        self.sensor=self.ns.CoincidentPeakPredictionSensor(None,'PJM RTO',150000,.8,
            'coincident_peak_prediction_system',Hass(),cover_mode='intermediate')
        finish(self.sensor._async_load_budget())
        self.sensor._last_load_update=Clock.now()
        self.sensor._current_prediction_date=Clock.now().date()

    def test_all_policy_paths_obey_cap(self):
        s=self.sensor
        s._compute_peak_hour_active=lambda now:setattr(s,'_peak_hour_active',True)
        s._check_peak_hour_active(Clock.now())
        self.assertTrue(s._peak_hour_active)
        Clock.value+=timedelta(hours=2)
        s._last_load_update=Clock.now()
        s._check_peak_hour_active(Clock.now())
        self.assertFalse(s._peak_hour_active)
        self.assertEqual(s._budget.seconds,7200)

    def test_exact_deadline_boundary_and_cleanup(self):
        s=self.sensor
        s._compute_peak_hour_active=lambda now:setattr(s,'_peak_hour_active',True)
        s._refresh_decision_hour=lambda now:None
        s._evaluate_5cp_risk=lambda:None
        finish(s.async_added_to_hass())
        s._check_peak_hour_active(Clock.now())
        live=[t for t in self.timers if not t['cancelled']]
        self.assertEqual({t['stamp'].astimezone(Clock.DEFAULT_TIME_ZONE).strftime('%H:%M') for t in live},
                         {'16:00','17:02'})
        Clock.value=datetime(2026,7,1,16,0,3,tzinfo=Clock.DEFAULT_TIME_ZONE)
        s._last_load_update=Clock.now()-timedelta(minutes=3)
        s._budget.active=False
        finish(s._async_dispatch_boundary(Clock.now().replace(second=0)))
        self.assertEqual(s._budget.last_activation,Clock.now().isoformat())
        finish(s.async_will_remove_from_hass())
        self.assertFalse(s._peak_hour_active)
        self.assertTrue(all(t['cancelled'] or t['action']==s._async_dispatch_boundary for t in self.timers))

    def test_stale_data_does_not_arm(self):
        s=self.sensor
        s._compute_peak_hour_active=lambda now:setattr(s,'_peak_hour_active',True)
        Clock.value+=timedelta(minutes=11)
        s._check_peak_hour_active(Clock.now())
        self.assertFalse(s._peak_hour_active)

    def test_reload_preserves_used_budget_and_first_arm(self):
        s=self.sensor
        s._cover5cp_armed_any=True; s._cover5cp_first_arm_h=15
        s._budget.advance(Clock.now(),True,7200)
        Clock.value+=timedelta(minutes=90)
        finish(s.async_will_remove_from_hass())
        restored=self.ns.CoincidentPeakPredictionSensor(None,'PJM RTO',150000,.8,
            'coincident_peak_prediction_system',Hass(),cover_mode='intermediate')
        restored._budget_store.data=s._budget_store.data
        finish(restored._async_load_budget())
        self.assertEqual(restored._budget.seconds,5400)
        self.assertEqual(restored._cover5cp_first_arm_h,15)


class PeakHistoryTests(unittest.TestCase):
    def setUp(self):
        self.ns,_=load_sensor()
        Clock.value=datetime(2026,10,1,12,tzinfo=Clock.DEFAULT_TIME_ZONE)
        self.peak=datetime(2026,7,2,16,tzinfo=Clock.DEFAULT_TIME_ZONE)
        self.saved=dict(top_five_peaks=[(self.peak.isoformat(),162703)],last_reset_date='2026-07-01')

    def sensor(self,data=None):
        obj=self.ns.CoincidentPeakPredictionSensor(None,'PJM RTO',150000,.8,
            'coincident_peak_prediction_system',Hass())
        obj._store.data=self.saved if data is None else data
        finish(obj._async_ensure_peaks_loaded())
        return obj

    def test_october_and_december_do_not_clear_peaks(self):
        s=self.sensor()
        for now in (Clock.now(),datetime(2026,12,31,23,59,tzinfo=Clock.DEFAULT_TIME_ZONE)):
            self.assertFalse(finish(s._async_reset_peaks_for_year(now)))
            self.assertEqual(s._top_five_peaks,[(self.peak,162703)])

    def test_reset_uses_local_midnight_not_utc(self):
        s=self.sensor()
        self.assertFalse(finish(s._async_reset_peaks_for_year(
            datetime(2027,1,1,5,59,tzinfo=timezone.utc))))
        self.assertTrue(finish(s._async_reset_peaks_for_year(
            datetime(2027,1,1,6,0,tzinfo=timezone.utc))))
        self.assertEqual(s._top_five_peaks,[])
        self.assertEqual(s._store.data['last_reset_date'],'2027-01-01')

    def test_restart_catches_up_and_then_preserves_new_year_board(self):
        Clock.value=datetime(2027,1,5,12,tzinfo=Clock.DEFAULT_TIME_ZONE)
        s=self.sensor()
        self.assertTrue(finish(s._async_reset_peaks_for_year(Clock.now())))
        current=datetime(2027,1,4,16,tzinfo=Clock.DEFAULT_TIME_ZONE)
        s._top_five_peaks=[(current,120000)]
        finish(s._async_save_peaks())
        restarted=self.sensor(s._store.data)
        self.assertFalse(finish(restarted._async_reset_peaks_for_year(Clock.now())))
        self.assertEqual(restarted._top_five_peaks,[(current,120000)])

    def test_missing_marker_infers_old_year_and_keeps_current_entries(self):
        Clock.value=datetime(2027,1,5,12,tzinfo=Clock.DEFAULT_TIME_ZONE)
        current=Clock.now()-timedelta(days=1)
        s=self.sensor(dict(top_five_peaks=[(self.peak.isoformat(),162703),(current.isoformat(),120000)]))
        self.assertTrue(finish(s._async_reset_peaks_for_year(Clock.now())))
        self.assertEqual(s._top_five_peaks,[(current,120000)])

    def test_storage_load_finishes_once_before_reset(self):
        Clock.value=datetime(2027,1,1,0,0,tzinfo=Clock.DEFAULT_TIME_ZONE)
        s=self.sensor()
        self.assertTrue(s._peaks_loaded)
        finish(s._async_reset_peaks_for_year(Clock.now()))
        s._store.data=self.saved  # An ensure call must not reload an old board.
        finish(s._async_ensure_peaks_loaded())
        self.assertEqual(s._top_five_peaks,[])

    def test_failed_save_does_not_suppress_retry(self):
        s=self.sensor()
        Clock.value=datetime(2027,1,1,0,0,tzinfo=Clock.DEFAULT_TIME_ZONE)
        save=s._store.async_save
        async def failed_save(data): raise OSError('synthetic storage failure')
        s._store.async_save=failed_save
        with self.assertRaises(OSError): finish(s._async_reset_peaks_for_year(Clock.now()))
        self.assertEqual(s._top_five_peaks,[(self.peak,162703)])
        s._store.async_save=save
        self.assertTrue(finish(s._async_reset_peaks_for_year(Clock.now())))


class PricesTests(unittest.TestCase):
    def setUp(self): self.prices=module('prices')

    def test_central_hour_and_fallback_preserved(self):
        items=[dict(datetime_beginning_utc='2026-07-01T21:00:00',total_lmp_da=70)]
        result=self.prices.parse_prices(items,local_timezone=Clock.DEFAULT_TIME_ZONE)
        self.assertEqual(result['by_hour'][16],70)
        items=[dict(datetime_beginning_utc=utc,total_lmp_da=price) for utc,price in
               [('2026-11-01T06:00:00',20),('2026-11-01T07:00:00',100)]]
        result=self.prices.parse_prices(items,local_timezone=Clock.DEFAULT_TIME_ZONE)
        self.assertEqual([r['price'] for r in result['intervals']],[20,100])
        self.assertIsNone(result['by_hour'][1])

    def test_eastern_only_ambiguous_hour_and_nan_rejected(self):
        self.assertIsNone(self.prices.interval_start(dict(datetime_beginning_ept='2026-11-01T01:00:00')))
        self.assertIsNone(self.prices.interval_start(dict(datetime_beginning_ept='2026-03-08T02:00:00')))
        self.assertEqual(self.prices.parse_prices([dict(datetime_beginning_utc='2026-07-01T21:00:00',total_lmp_da='nan')])['intervals'],[])

    def test_failed_new_day_fetch_retains_last_price(self):
        ns,_=load_sensor()
        Clock.value=datetime(2026,7,1,16,30,tzinfo=Clock.DEFAULT_TIME_ZONE)
        class API:
            async def async_update_da_hrl_lmp(self,zone): return None
        s=ns.PJMSensor(API(),'da_hrl_lmp','COMED',None)
        s._da_hrl_day=Clock.now().date()-timedelta(days=1)
        s._da_hrl_by_hour={h:10 for h in range(24)}
        s._state=10
        finish(s.update_da_hrl_lmp())
        self.assertEqual(s._state,10)
        self.assertTrue(s._attr_available)
        self.assertTrue(s._da_hrl_stale)

    def test_local_day_dst_utc_query(self):
        ns,_=load_sensor()
        Clock.value=datetime(2026,11,1,12,tzinfo=Clock.DEFAULT_TIME_ZONE)
        api=ns.PJMData(None,'synthetic')
        calls=[]
        async def no_wait(): pass
        async def fetch(**kwargs): calls.append(kwargs); return []
        api._rate_limit=no_wait; api._fetch_da_hrl_lmp_items=fetch
        finish(api.async_update_da_hrl_lmp('COMED'))
        self.assertEqual(calls[0]['time_string'],'11/01/2026 05:00:00to11/02/2026 05:59:59')


class NetworkTests(unittest.TestCase):
    def test_redirect_timeout_and_response_cleanup(self):
        ns,_=load_sensor()
        class Response:
            def __init__(self,status=200): self.status=status; self.closed=False
            async def __aenter__(self): return self
            async def __aexit__(self,*args): self.closed=True
            def raise_for_status(self):
                if self.status>=400: raise ClientError('HTTP error')
            async def json(self): return {'items':[{'area':'PJM RTO','instantaneous_load':150000}]}
        class Session:
            def __init__(self,status): self.status=status; self.calls=[]; self.responses=[]
            def get(self,url,**kwargs):
                self.calls.append(kwargs); response=Response(self.status); self.responses.append(response)
                return response
        async def no_wait(*args): pass
        for status in (200,302,429):
            session=Session(status); api=ns.PJMData(session,'synthetic-never-sent')
            api._rate_limit=no_wait
            with patch('asyncio.sleep',no_wait): result=finish(api.async_update_instantaneous('PJM RTO'))
            self.assertEqual(result,150000 if status==200 else None)
            self.assertTrue(all(r.closed for r in session.responses))
            self.assertTrue(all(c['allow_redirects'] is False and c['timeout'].total==60 for c in session.calls))
            self.assertEqual(len(session.calls),1 if status==200 else 3)

    def test_body_timeout_retries_and_releases_response(self):
        ns,_=load_sensor()
        closed=[]
        class Response:
            status=200
            async def __aenter__(self): return self
            async def __aexit__(self,*args): closed.append(True)
            def raise_for_status(self): pass
            async def json(self): raise TimeoutError('synthetic timeout')
        session=SimpleNamespace(get=lambda *args,**kwargs:Response())
        api=ns.PJMData(session,'synthetic')
        async def no_wait(*args): pass
        api._rate_limit=no_wait
        with patch('asyncio.sleep',no_wait):
            self.assertIsNone(finish(api.async_update_instantaneous('PJM RTO')))
        self.assertEqual(len(closed),3)

    def test_broken_json_does_not_leak_response(self):
        ns,_=load_sensor()
        closed=[]
        class Response:
            status=200
            async def __aenter__(self): return self
            async def __aexit__(self,*args): closed.append(True)
            def raise_for_status(self): pass
            async def json(self): raise ValueError('synthetic invalid JSON')
        api=ns.PJMData(SimpleNamespace(get=lambda *args,**kwargs:Response()),'synthetic')
        async def no_wait(*args): pass
        api._rate_limit=no_wait
        self.assertIsNone(finish(api.async_update_instantaneous('PJM RTO')))
        self.assertEqual(len(closed),1)


class ReleaseTests(unittest.TestCase):
    def test_metadata_and_migration_reload(self):
        manifest=json.loads((SOURCE/'manifest.json').read_text())
        self.assertEqual(manifest['version'],'2.3.1')
        self.assertEqual(manifest['iot_class'],'cloud_polling')
        tree=ast.parse((SOURCE/'__init__.py').read_text())
        constants=module('const')
        calls=[]
        async def reload(entry_id): calls.append(entry_id)
        def update(entry,**kwargs):
            for key,value in kwargs.items(): setattr(entry,key,value)
        ns=dict(HomeAssistant=object,ConfigEntry=object,
                _LOGGER=__import__('logging').getLogger('test'),
                CONF_COVER_MODE=constants.CONF_COVER_MODE,
                CONF_EXPERIMENTAL_COINCIDENT=constants.CONF_EXPERIMENTAL_COINCIDENT,
                DEFAULT_COVER_MODE=constants.DEFAULT_COVER_MODE)
        nodes=[n for n in tree.body if isinstance(n,ast.AsyncFunctionDef)
               and n.name in ('async_migrate_entry','_async_update_options')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'setup.py','exec'),ns)
        hass=SimpleNamespace(config_entries=SimpleNamespace(async_update_entry=update,async_reload=reload))
        entry=SimpleNamespace(entry_id='test',version=1,data={'zone':'COMED','api_key':'synthetic','sensors':['instantaneous_total_load']})
        self.assertTrue(finish(ns['async_migrate_entry'](hass,entry)))
        self.assertEqual(entry.version,2)
        self.assertEqual(entry.data['api_key'],'synthetic')
        self.assertEqual(entry.data['cover_mode'],'secure')
        finish(ns['_async_update_options'](hass,entry))
        self.assertEqual(calls,['test'])


if __name__=='__main__': unittest.main()
