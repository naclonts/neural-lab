"""Runnable calibration candidate; prospective independent review required before queueing."""

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HOME'] = '/output/calibration/hf-cache'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
import base64, hashlib, json, math, platform, resource, shutil, subprocess, sys, time, traceback, types
from pathlib import Path
import numpy as np
import torch
import importlib.util
HELPER_SHA = '6ef5863486d3f9cfa4fac8b06823ff399986c78e898130e979a9035c6f062baf'
helper_spec = importlib.util.find_spec('neural_lab.artifacts')
if helper_spec is None or helper_spec.origin is None or hashlib.sha256(Path(helper_spec.origin).read_bytes()).hexdigest() != HELPER_SHA:
    raise RuntimeError('artifact helper hash mismatch')
from torch.nn import functional as F
from neural_lab.artifacts import ArtifactWriter, BestCheckpoint, load_json_state
import neural_lab.artifacts as artifacts

EMBEDDED_SOURCE = (
    'IiIiUFJPUE9TRUQgT05MWTogcmV2aWV3IGFuZCBmcmVlemUgYmVmb3JlIGFueSBleGVjdXRpb24gb3IgcXVldWUgc3VibWlzc2lv'
    'bi4iIiIKaW1wb3J0IG9zCm9zLmVudmlyb25bJ0NVQkxBU19XT1JLU1BBQ0VfQ09ORklHJ10gPSAnOjQwOTY6OCcKb3MuZW52aXJv'
    'blsnSEZfSE9NRSddID0gJy9vdXRwdXQvaGYtY2FjaGUnCm9zLmVudmlyb25bJ0hGX0hVQl9ESVNBQkxFX0lNUExJQ0lUX1RPS0VO'
    'J10gPSAnMScKaW1wb3J0IGNvcHkKaW1wb3J0IGhhc2hsaWIKaW1wb3J0IGpzb24KaW1wb3J0IG1hdGgKZnJvbSBwYXRobGliIGlt'
    'cG9ydCBQYXRoCmltcG9ydCB0aW1lCmltcG9ydCB0cmFjZWJhY2sKaW1wb3J0IHBsYXRmb3JtCmltcG9ydCBzaHV0aWwKaW1wb3J0'
    'IHN5cwppbXBvcnQgbnVtcHkgYXMgbnAKaW1wb3J0IHRvcmNoCmZyb20gdG9yY2ggaW1wb3J0IG5uCmZyb20gdG9yY2gubm4gaW1w'
    'b3J0IGZ1bmN0aW9uYWwgYXMgRgpmcm9tIG5ldXJhbF9sYWIuYXJ0aWZhY3RzIGltcG9ydCBBcnRpZmFjdFdyaXRlciwgQmVzdENo'
    'ZWNrcG9pbnQKCiMgRGVsaWJlcmF0ZSBzdWJtaXNzaW9uIGdhdGU6IHRoaXMgaXMgYSBwcm9wb3NlZCBpbXBsZW1lbnRhdGlvbiwg'
    'bm90IGEgZnJvemVuIHByb3RvY29sLgpSRVZJRVdFRF9BTkRfRlJPWkVOID0gRmFsc2UKQ0ZHID0gZGljdChlcG9jaHM9MTAwLCBz'
    'ZWVkcz1bNDgwMSw0ODAyLDQ4MDMsNDgwNCw0ODA1LDQ4MDZdLCBzcGxpdF9zZWVkPTIwMjYxMTAxLAogICAgICAgICAgIGJhdGNo'
    'PTI1NiwgbHI9LjAwMSwgd2VpZ2h0X2RlY2F5PS4wMSwgdGhyZXNob2xkPS0uMDEpCkFSTVMgPSAoJ2R5bmFtaWMnLCdzdGF0aWMn'
    'LCdzaW5nbGUnKQpPVVQgPSBQYXRoKCcvb3V0cHV0JykKUkVWID0gJ2FhZGIzYWY3N2U5MDQ4YWRiZWE2YjQ3YzIxYTgxZTQ3ZGQw'
    'OTJhZTUnCkZJTEUgPSAnY2lmYXIxMDAvdHJhaW4tMDAwMDAtb2YtMDAwMDEucGFycXVldCcKU0hBID0gJzY5NDg2NWQ2Yjk5MGUy'
    'MzQ4MDRmMDEyNjg1ODZjNDFlODhiY2JiYjc1ZTIwODU4NDMyYzA1YWQ0MDgxYWNhMjMnClNUQVJUID0gdGltZS5tb25vdG9uaWMo'
    'KQoKY2xhc3MgQ29uZGl0aW9uYWxDb252KG5uLk1vZHVsZSk6CiAgICBkZWYgX19pbml0X18oc2VsZiwgc3RhdGljKToKICAgICAg'
    'ICBzdXBlcigpLl9faW5pdF9fKCkKICAgICAgICBzZWxmLnN0YXRpYyA9IHN0YXRpYwogICAgICAgIHNlbGYuZXhwZXJ0cyA9IG5u'
    'Lk1vZHVsZUxpc3QoW25uLkNvbnYyZCg2NCw2NCwzLHBhZGRpbmc9MSkgZm9yIF8gaW4gcmFuZ2UoMildKQogICAgICAgIHNlbGYu'
    'Z2F0ZSA9IG5uLlNlcXVlbnRpYWwobm4uTGluZWFyKDY0LDE2KSxubi5HRUxVKCksbm4uTGluZWFyKDE2LDIpKQogICAgZGVmIGZv'
    'cndhcmQoc2VsZiwgeCk6CiAgICAgICAgeiA9IHgubWVhbigoMiwzKSkKICAgICAgICBpZiBzZWxmLnN0YXRpYzoKICAgICAgICAg'
    'ICAgeiA9IHRvcmNoLnplcm9zX2xpa2UoeikKICAgICAgICBwID0gc2VsZi5nYXRlKHopLnNvZnRtYXgoZGltPTEpCiAgICAgICAg'
    'ZSA9IHRvcmNoLnN0YWNrKFtjb252KHgpIGZvciBjb252IGluIHNlbGYuZXhwZXJ0c10sZGltPTEpCiAgICAgICAgcmV0dXJuIChl'
    'ICogcFs6LDosTm9uZSxOb25lLE5vbmVdKS5zdW0oZGltPTEpCgpjbGFzcyBNb2RlbChubi5Nb2R1bGUpOgogICAgZGVmIF9faW5p'
    'dF9fKHNlbGYsIGFybSk6CiAgICAgICAgc3VwZXIoKS5fX2luaXRfXygpCiAgICAgICAgaWYgYXJtIG5vdCBpbiBBUk1TOiByYWlz'
    'ZSBWYWx1ZUVycm9yKCd1bmtub3duIGFybScpCiAgICAgICAgc2VsZi5zdGVtID0gbm4uU2VxdWVudGlhbChubi5Db252MmQoMyw2'
    'NCwyLHN0cmlkZT0yKSxubi5HRUxVKCksbm4uQmF0Y2hOb3JtMmQoNjQpKQogICAgICAgIHNlbGYuYmxvY2tzID0gbm4uTW9kdWxl'
    'TGlzdChbbm4uU2VxdWVudGlhbChubi5Db252MmQoNjQsNjQsMyxwYWRkaW5nPTEpLG5uLkdFTFUoKSxubi5CYXRjaE5vcm0yZCg2'
    'NCkpIGZvciBfIGluIHJhbmdlKDMpXSkKICAgICAgICBzZWxmLmxhc3QgPSBubi5TZXF1ZW50aWFsKG5uLkNvbnYyZCg2NCw2NCwz'
    'LHBhZGRpbmc9MSkgaWYgYXJtID09ICdzaW5nbGUnIGVsc2UgQ29uZGl0aW9uYWxDb252KGFybSA9PSAnc3RhdGljJyksbm4uR0VM'
    'VSgpLG5uLkJhdGNoTm9ybTJkKDY0KSkKICAgICAgICBzZWxmLmhlYWQgPSBubi5MaW5lYXIoNjQsMTAwKQogICAgZGVmIGZvcndh'
    'cmQoc2VsZix4KToKICAgICAgICB4ID0gc2VsZi5zdGVtKHgpCiAgICAgICAgZm9yIGJsb2NrIGluIHNlbGYuYmxvY2tzOiB4ID0g'
    'YmxvY2soeCkKICAgICAgICByZXR1cm4gc2VsZi5oZWFkKHNlbGYubGFzdCh4KS5tZWFuKCgyLDMpKSkKCmRlZiBwYWlyZWRfaW5p'
    'dGlhbF9zdGF0ZXMoc2VlZCk6CiAgICB0b3JjaC5tYW51YWxfc2VlZChzZWVkKQogICAgc2luZ2xlPU1vZGVsKCdzaW5nbGUnKTsg'
    'ZHluYW1pYz1Nb2RlbCgnZHluYW1pYycpCiAgICB3aXRoIHRvcmNoLm5vX2dyYWQoKToKICAgICAgICByZWZlcmVuY2U9c2luZ2xl'
    'Lmxhc3RbMF0KICAgICAgICBwZXJ0dXJiPXRvcmNoLnJhbmRuX2xpa2UocmVmZXJlbmNlLndlaWdodCkqKC4wMSpyZWZlcmVuY2Uu'
    'd2VpZ2h0LnN0ZCh1bmJpYXNlZD1GYWxzZSkpCiAgICAgICAgZHluYW1pYy5sYXN0WzBdLmV4cGVydHNbMF0ud2VpZ2h0LmNvcHlf'
    'KHJlZmVyZW5jZS53ZWlnaHQrcGVydHVyYikKICAgICAgICBkeW5hbWljLmxhc3RbMF0uZXhwZXJ0c1sxXS53ZWlnaHQuY29weV8o'
    'cmVmZXJlbmNlLndlaWdodC1wZXJ0dXJiKQogICAgICAgIGZvciBleHBlcnQgaW4gZHluYW1pYy5sYXN0WzBdLmV4cGVydHM6IGV4'
    'cGVydC5iaWFzLmNvcHlfKHJlZmVyZW5jZS5iaWFzKQogICAgICAgIGR5bmFtaWMubGFzdFswXS5nYXRlWy0xXS53ZWlnaHQuemVy'
    'b18oKTsgZHluYW1pYy5sYXN0WzBdLmdhdGVbLTFdLmJpYXMuemVyb18oKQogICAgICAgIGR5bmFtaWMuc3RlbS5sb2FkX3N0YXRl'
    'X2RpY3Qoc2luZ2xlLnN0ZW0uc3RhdGVfZGljdCgpKTsgZHluYW1pYy5ibG9ja3MubG9hZF9zdGF0ZV9kaWN0KHNpbmdsZS5ibG9j'
    'a3Muc3RhdGVfZGljdCgpKQogICAgICAgIGR5bmFtaWMubGFzdFsyXS5sb2FkX3N0YXRlX2RpY3Qoc2luZ2xlLmxhc3RbMl0uc3Rh'
    'dGVfZGljdCgpKTsgZHluYW1pYy5oZWFkLmxvYWRfc3RhdGVfZGljdChzaW5nbGUuaGVhZC5zdGF0ZV9kaWN0KCkpCiAgICByZXR1'
    'cm4gY29weS5kZWVwY29weShzaW5nbGUuc3RhdGVfZGljdCgpKSwgY29weS5kZWVwY29weShkeW5hbWljLnN0YXRlX2RpY3QoKSkK'
    'CmRlZiBwYWlyZWRfc3VtbWFyeSh2YWx1ZXMpOgogICAgaWYgbGVuKHZhbHVlcykhPWxlbihDRkdbJ3NlZWRzJ10pIG9yIG5vdCBh'
    'bGwobWF0aC5pc2Zpbml0ZSh2KSBmb3IgdiBpbiB2YWx1ZXMpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ2FsbCBmaW5pdGUg'
    'cGFpcmVkIHZhbHVlcyByZXF1aXJlZCcpCiAgICBtZWFuPXN1bSh2YWx1ZXMpL2xlbih2YWx1ZXMpCiAgICBzZD1tYXRoLnNxcnQo'
    'c3VtKCh4LW1lYW4pKioyIGZvciB4IGluIHZhbHVlcykvKGxlbih2YWx1ZXMpLTEpKQogICAgbWFyZ2luPTIuNTcwNTgxODM1NjM2'
    'MzE0KnNkL21hdGguc3FydChsZW4odmFsdWVzKSkKICAgIHJldHVybiBkaWN0KHZhbHVlcz12YWx1ZXMsbWVhbj1tZWFuLHNhbXBs'
    'ZV9zZD1zZCxkZXNjcmlwdGl2ZV90OTU9W21lYW4tbWFyZ2luLG1lYW4rbWFyZ2luXSkKCkB0b3JjaC5ub19ncmFkKCkKZGVmIHJv'
    'dXRpbmdfZGlhZ25vc3RpY3MobW9kZWwscmF3LGlkcyxtdSxzZCk6CiAgICBpZiBub3QgaXNpbnN0YW5jZShtb2RlbC5sYXN0WzBd'
    'LCBDb25kaXRpb25hbENvbnYpOiByZXR1cm4gTm9uZQogICAgbW9kZWwuZXZhbCgpOyBsYXllcj1tb2RlbC5sYXN0WzBdOyB3ZWln'
    'aHRzPVtdCiAgICBkZWYgY2FwdHVyZShtb2R1bGUsYXJncyk6CiAgICAgICAgeD1hcmdzWzBdOyB6PXgubWVhbigoMiwzKSkKICAg'
    'ICAgICBpZiBsYXllci5zdGF0aWM6IHo9dG9yY2guemVyb3NfbGlrZSh6KQogICAgICAgIHdlaWdodHMuYXBwZW5kKGxheWVyLmdh'
    'dGUoeikuc29mdG1heCgxKS5jcHUoKSkKICAgIGhvb2s9bGF5ZXIucmVnaXN0ZXJfZm9yd2FyZF9wcmVfaG9vayhjYXB0dXJlKQog'
    'ICAgdHJ5OgogICAgICAgIGZvciBzdGFydCBpbiByYW5nZSgwLGxlbihpZHMpLENGR1snYmF0Y2gnXSk6IG1vZGVsKGlucHV0cyhy'
    'YXcsaWRzW3N0YXJ0OnN0YXJ0K0NGR1snYmF0Y2gnXV0sbXUsc2QpKQogICAgZmluYWxseTogaG9vay5yZW1vdmUoKQogICAgcD10'
    'b3JjaC5jYXQod2VpZ2h0cywwKTsgZW50cm9weT0tKHAqcC5jbGFtcF9taW4oMWUtMzApLmxvZygpKS5zdW0oMSkKICAgIHJldHVy'
    'biBkaWN0KGV4YW1wbGVzPWxlbihwKSxtZWFuX3dlaWdodHM9cC5tZWFuKDApLnRvbGlzdCgpLHdlaWdodHNfcG9wdWxhdGlvbl9z'
    'dGQ9cC5zdGQoMCx1bmJpYXNlZD1GYWxzZSkudG9saXN0KCksbWVhbl9lbnRyb3B5PWZsb2F0KGVudHJvcHkubWVhbigpKSxleHBl'
    'cnRfd2VpZ2h0X2wyPWZsb2F0KHRvcmNoLmxpbmFsZy52ZWN0b3Jfbm9ybShsYXllci5leHBlcnRzWzBdLndlaWdodC1sYXllci5l'
    'eHBlcnRzWzFdLndlaWdodCkpLGV4cGVydF9iaWFzX2wyPWZsb2F0KHRvcmNoLmxpbmFsZy52ZWN0b3Jfbm9ybShsYXllci5leHBl'
    'cnRzWzBdLmJpYXMtbGF5ZXIuZXhwZXJ0c1sxXS5iaWFzKSkpCgpkZWYgc2NoZWR1bGUoc2VlZCwgZXBvY2hzLCBuKToKICAgIG9y'
    'ZGVyLCBjcm9wLCBmbGlwID0gW10sIFtdLCBbXQogICAgZm9yIGVwb2NoIGluIHJhbmdlKGVwb2Nocyk6CiAgICAgICAgcm5nID0g'
    'bnAucmFuZG9tLmRlZmF1bHRfcm5nKHNlZWQrMTAwMDAwK2Vwb2NoKjEwMDAwMDMpCiAgICAgICAgb3JkZXIuYXBwZW5kKHJuZy5w'
    'ZXJtdXRhdGlvbihuKS5hc3R5cGUobnAuaW50MzIpKQogICAgICAgIGNyb3AuYXBwZW5kKHJuZy5pbnRlZ2VycygwLDksKG4sMiks'
    'ZHR5cGU9bnAuaW50MTYpKQogICAgICAgIGZsaXAuYXBwZW5kKHJuZy5pbnRlZ2VycygwLDIsbixkdHlwZT1ucC51aW50OCkpCiAg'
    'ICByZXR1cm4gbnAuc3RhY2sob3JkZXIpLG5wLnN0YWNrKGNyb3ApLG5wLnN0YWNrKGZsaXApCgpkZWYgaW5wdXRzKHJhdywgaWRz'
    'LCBtZWFuLCBzdGQsIGNyb3A9Tm9uZSwgZmxpcD1Ob25lKToKICAgIHggPSByYXdbaWRzXS5mbG9hdCgpLmRpdl8oMjU1KQogICAg'
    'aWYgY3JvcCBpcyBub3QgTm9uZToKICAgICAgICB4ID0gRi5wYWQoeCwoNCw0LDQsNCksbW9kZT0ncmVmbGVjdCcpCiAgICAgICAg'
    'cm93cyA9IHRvcmNoLmFyYW5nZSgzMixkZXZpY2U9eC5kZXZpY2UpW05vbmUsOl0rY3JvcFs6LDAsTm9uZV0KICAgICAgICBjb2xz'
    'ID0gdG9yY2guYXJhbmdlKDMyLGRldmljZT14LmRldmljZSlbTm9uZSw6XStjcm9wWzosMSxOb25lXQogICAgICAgIHggPSB4W3Rv'
    'cmNoLmFyYW5nZShsZW4oaWRzKSxkZXZpY2U9eC5kZXZpY2UpWzosTm9uZSxOb25lXSw6LHJvd3NbOiw6LE5vbmVdLGNvbHNbOixO'
    'b25lLDpdXS5wZXJtdXRlKDAsMywxLDIpCiAgICAgICAgeCA9IHRvcmNoLndoZXJlKGZsaXBbOixOb25lLE5vbmUsTm9uZV0uYm9v'
    'bCgpLHguZmxpcCgtMSkseCkKICAgIHJldHVybiAoeC1tZWFuKS9zdGQKCkB0b3JjaC5ub19ncmFkKCkKZGVmIGV2YWx1YXRlKG1v'
    'ZGVsLHJhdyx5LGlkcyxtZWFuLHN0ZCk6CiAgICBtb2RlbC5ldmFsKCk7IGxvc3M9MC47IGNvcnJlY3Q9MDsgb3V0cHV0cz1bXQog'
    'ICAgZm9yIHN0YXJ0IGluIHJhbmdlKDAsbGVuKGlkcyksQ0ZHWydiYXRjaCddKToKICAgICAgICBpeD1pZHNbc3RhcnQ6c3RhcnQr'
    'Q0ZHWydiYXRjaCddXTsgbG9naXRzPW1vZGVsKGlucHV0cyhyYXcsaXgsbWVhbixzdGQpKQogICAgICAgIG91dHB1dHMuYXBwZW5k'
    'KGxvZ2l0cy5jcHUoKS5udW1weSgpKTsgbG9zcyArPSBGLmNyb3NzX2VudHJvcHkobG9naXRzLHlbaXhdLHJlZHVjdGlvbj0nc3Vt'
    'JykuaXRlbSgpCiAgICAgICAgY29ycmVjdCArPSAobG9naXRzLmFyZ21heCgxKT09eVtpeF0pLnN1bSgpLml0ZW0oKQogICAgcmV0'
    'dXJuIGRpY3QoY2U9bG9zcy9sZW4oaWRzKSxhY2N1cmFjeT1jb3JyZWN0L2xlbihpZHMpKSwgbnAuY29uY2F0ZW5hdGUob3V0cHV0'
    'cykKCmRlZiBldmVudChraW5kLCoqa3cpOgogICAgcm93PWRpY3QoZXZlbnQ9a2luZCxlbGFwc2VkX3NlY29uZHM9dGltZS5tb25v'
    'dG9uaWMoKS1TVEFSVCwqKmt3KQogICAgd2l0aCAoT1VULydwcm9ncmVzcy5qc29ubCcpLm9wZW4oJ2EnKSBhcyBmOiBmLndyaXRl'
    'KGpzb24uZHVtcHMocm93LGFsbG93X25hbj1GYWxzZSkrJ1xuJykKICAgIHByaW50KGpzb24uZHVtcHMocm93KSxmbHVzaD1UcnVl'
    'KQoKZGVmIG1haW4oKToKICAgIGlmIG5vdCBSRVZJRVdFRF9BTkRfRlJPWkVOOiByYWlzZSBSdW50aW1lRXJyb3IoJ1Byb3Bvc2Vk'
    'IGNhbmRpZGF0ZSBtdXN0IGJlIGluZGVwZW5kZW50bHkgcmV2aWV3ZWQgYW5kIGZyb3plbiBiZWZvcmUgZXhlY3V0aW9uJykKICAg'
    'IGlmIG5vdCB0b3JjaC5jdWRhLmlzX2F2YWlsYWJsZSgpOiByYWlzZSBSdW50aW1lRXJyb3IoJ0NVREEgcmVxdWlyZWQ7IG5vIENQ'
    'VSBmYWxsYmFjaycpCiAgICB0b3JjaC5zZXRfbnVtX3RocmVhZHMoNCk7IHRvcmNoLnVzZV9kZXRlcm1pbmlzdGljX2FsZ29yaXRo'
    'bXMoVHJ1ZSkKICAgIHRvcmNoLmJhY2tlbmRzLmN1ZG5uLmJlbmNobWFyaz1GYWxzZTsgdG9yY2guYmFja2VuZHMuY3VkYS5tYXRt'
    'dWwuYWxsb3dfdGYzMj1GYWxzZTsgdG9yY2guYmFja2VuZHMuY3Vkbm4uYWxsb3dfdGYzMj1GYWxzZQogICAgT1VULm1rZGlyKHBh'
    'cmVudHM9VHJ1ZSxleGlzdF9vaz1UcnVlKQogICAgaWYgKE9VVC8nYXJ0aWZhY3RzLmpzb24nKS5leGlzdHMoKSBvciAoT1VULydw'
    'cm9ncmVzcy5qc29ubCcpLmV4aXN0cygpOiByYWlzZSBSdW50aW1lRXJyb3IoJ2ZyZXNoIG91dHB1dCBkaXJlY3RvcnkgcmVxdWly'
    'ZWQnKQogICAgd3JpdGVyPUFydGlmYWN0V3JpdGVyKE9VVCkKICAgIGltcG9ydCBuZXVyYWxfbGFiLmFydGlmYWN0cyBhcyBhcnRp'
    'ZmFjdHMKICAgIChPVVQvJ3Byb3ZlbmFuY2UnKS5ta2RpcigpCiAgICBmb3Igc3JjLG5hbWUgaW4gKChQYXRoKF9fZmlsZV9fKSwn'
    'ZXhwZXJpbWVudC5weScpLChQYXRoKGFydGlmYWN0cy5fX2ZpbGVfXyksJ2FydGlmYWN0cy5weScpKToKICAgICAgICBkc3Q9T1VU'
    'Lydwcm92ZW5hbmNlJy9uYW1lOyBzaHV0aWwuY29weWZpbGUoc3JjLGRzdCk7IHdyaXRlci5yZWdpc3Rlcl9maWxlKCdkYXRhc2V0'
    'cycsZHN0LG5hbWUsZm9ybWF0PSdweXRob24nLHNvdXJjZT1kaWN0KHJvbGU9J2V4ZWN1dGVkIHNvdXJjZSBzbmFwc2hvdCcpKQog'
    'ICAgZnJvbSBodWdnaW5nZmFjZV9odWIgaW1wb3J0IEhmQXBpLGhmX2h1Yl9kb3dubG9hZAogICAgaW1wb3J0IGh1Z2dpbmdmYWNl'
    'X2h1YiwgZGF0YXNldHMKICAgIGluZm89SGZBcGkodG9rZW49RmFsc2UpLmRhdGFzZXRfaW5mbygndW9mdC1jcy9jaWZhcjEwMCcs'
    'cmV2aXNpb249UkVWLHRpbWVvdXQ9MjApCiAgICBpZiBpbmZvLnNoYSAhPSBSRVY6IHJhaXNlIFZhbHVlRXJyb3IoJ2RhdGFzZXQg'
    'cmV2aXNpb24gbWlzbWF0Y2gnKQogICAgcGF0aD1oZl9odWJfZG93bmxvYWQocmVwb19pZD0ndW9mdC1jcy9jaWZhcjEwMCcscmVw'
    'b190eXBlPSdkYXRhc2V0JyxmaWxlbmFtZT1GSUxFLHJldmlzaW9uPVJFVix0b2tlbj1GYWxzZSxsb2NhbF9kaXI9c3RyKE9VVC8n'
    'aGYtc291cmNlJykpCiAgICBpZiBoYXNobGliLnNoYTI1NihQYXRoKHBhdGgpLnJlYWRfYnl0ZXMoKSkuaGV4ZGlnZXN0KCkhPVNI'
    'QTogcmFpc2UgVmFsdWVFcnJvcignZGF0YXNldCBoYXNoIG1pc21hdGNoJykKICAgIHNvdXJjZT13cml0ZXIucmVnaXN0ZXJfZmls'
    'ZSgnZGF0YXNldHMnLHBhdGgsJ2NpZmFyMTAwLW9mZmljaWFsLXRyYWluaW5nLXNvdXJjZScsZm9ybWF0PSdwYXJxdWV0Jyxzb3Vy'
    'Y2U9ZGljdChyZXBvPSd1b2Z0LWNzL2NpZmFyMTAwJyxyZXZpc2lvbj1SRVYsZmlsZT1GSUxFLHNoYTI1Nj1TSEEsY3JlYXRvcj0n'
    'aHR0cHM6Ly93d3cuY3MudG9yb250by5lZHUvfmtyaXovY2lmYXIuaHRtbCcsbGljZW5zZT1pbmZvLmNhcmRfZGF0YS5nZXQoJ2xp'
    'Y2Vuc2UnKSBpZiBpbmZvLmNhcmRfZGF0YSBlbHNlIE5vbmUsaHVnZ2luZ2ZhY2VfaHViX3ZlcnNpb249aHVnZ2luZ2ZhY2VfaHVi'
    'Ll9fdmVyc2lvbl9fLGRhdGFzZXRzX3ZlcnNpb249ZGF0YXNldHMuX192ZXJzaW9uX18pKQogICAgZHM9ZGF0YXNldHMubG9hZF9k'
    'YXRhc2V0KCdwYXJxdWV0JyxkYXRhX2ZpbGVzPXsndHJhaW4nOnBhdGh9LHNwbGl0PSd0cmFpbicsY2FjaGVfZGlyPXN0cihPVVQv'
    'J2hmLWNhY2hlJy8nZGVjb2RlZCcpKQogICAgaW1hZ2VzPW5wLnN0YWNrKFtucC5hc2FycmF5KHJvd1snaW1nJ10uY29udmVydCgn'
    'UkdCJyksZHR5cGU9bnAudWludDgpIGZvciByb3cgaW4gZHNdKTsgbGFiZWxzPW5wLmFzYXJyYXkoZHNbJ2ZpbmVfbGFiZWwnXSxk'
    'dHlwZT1ucC5pbnQ2NCkKICAgIGlmIGltYWdlcy5zaGFwZSE9KDUwMDAwLDMyLDMyLDMpIG9yIG5vdCBucC5hcnJheV9lcXVhbChu'
    'cC5iaW5jb3VudChsYWJlbHMsbWlubGVuZ3RoPTEwMCksbnAuZnVsbCgxMDAsNTAwKSk6IHJhaXNlIFZhbHVlRXJyb3IoJ3VuZXhw'
    'ZWN0ZWQgQ0lGQVItMTAwIGxheW91dCBvciBmaW5lLWxhYmVsIGNvdW50cycpCiAgICBybmc9bnAucmFuZG9tLmRlZmF1bHRfcm5n'
    'KENGR1snc3BsaXRfc2VlZCddKTsgdHJhaW49W107IGRldj1bXQogICAgZm9yIGMgaW4gcmFuZ2UoMTAwKToKICAgICAgICBpZHM9'
    'cm5nLnBlcm11dGF0aW9uKG5wLmZsYXRub256ZXJvKGxhYmVscz09YykpOyB0cmFpbi5leHRlbmQoaWRzWzo0NTBdKTsgZGV2LmV4'
    'dGVuZChpZHNbNDUwOl0pCiAgICB0cmFpbj1ucC5hc2FycmF5KHNvcnRlZCh0cmFpbikpOyBkZXY9bnAuYXNhcnJheShzb3J0ZWQo'
    'ZGV2KSkKICAgIGlmIGxlbih0cmFpbikhPTQ1MDAwIG9yIGxlbihkZXYpIT01MDAwIG9yIGxlbihucC51bmlxdWUobnAuY29uY2F0'
    'ZW5hdGUoW3RyYWluLGRldl0pKSkhPTUwMDAwIG9yIG5wLmludGVyc2VjdDFkKHRyYWluLGRldikuc2l6ZSBvciBub3QgbnAuYXJy'
    'YXlfZXF1YWwobnAuYmluY291bnQobGFiZWxzW3RyYWluXSxtaW5sZW5ndGg9MTAwKSxucC5mdWxsKDEwMCw0NTApKSBvciBub3Qg'
    'bnAuYXJyYXlfZXF1YWwobnAuYmluY291bnQobGFiZWxzW2Rldl0sbWlubGVuZ3RoPTEwMCksbnAuZnVsbCgxMDAsNTApKTogcmFp'
    'c2UgVmFsdWVFcnJvcignaW52YWxpZCBzcGxpdCcpCiAgICBkZWNvZGVkX3NoYT1oYXNobGliLnNoYTI1NihpbWFnZXMudG9ieXRl'
    'cygpK2xhYmVscy50b2J5dGVzKCkpLmhleGRpZ2VzdCgpCiAgICBwaXhlbHM9aW1hZ2VzW3RyYWluXS5hc3R5cGUobnAuZmxvYXQ2'
    'NCkvMjU1OyBtZWFuPXBpeGVscy5tZWFuKCgwLDEsMikpOyBzdGQ9cGl4ZWxzLnN0ZCgoMCwxLDIpKTsgZGVsIHBpeGVscwogICAg'
    'ZGF0YT13cml0ZXIuc2F2ZV9kYXRhc2V0KCdjaWZhcjEwMC10cmFpbmluZy1zcGxpdCcsZGljdChpbWFnZXM9aW1hZ2VzLGxhYmVs'
    'cz1sYWJlbHMsdHJhaW5faWRzPXRyYWluLGRldl9pZHM9ZGV2LG1lYW49bWVhbixzdGQ9c3RkKSxwcm92ZW5hbmNlPWRpY3Qoc291'
    'cmNlPXNvdXJjZSxzcGxpdF9zZWVkPUNGR1snc3BsaXRfc2VlZCddLGdlbmVyYXRvcj0nbnVtcHkgUENHNjQsIGNsYXNzIG9yZGVy'
    'IDAuLjk5LCBmaXJzdCA0NTAgcGVyIGNsYXNzIHRyYWluLCByZXN0IGRldmVsb3BtZW50LCBzb3J0ZWQgSURzJyxudW1weV92ZXJz'
    'aW9uPW5wLl9fdmVyc2lvbl9fLG9mZmljaWFsX3Rlc3Q9J25ldmVyIHJlcXVlc3RlZCcsZGVjb2RlZF9zaGEyNTZfaW1hZ2VzX3Ro'
    'ZW5fbGFiZWxzPWRlY29kZWRfc2hhLG5vcm1hbGl6YXRpb249J3RyYWluaW5nIFJHQiBwb3B1bGF0aW9uIG1lYW4gYW5kIHN0YW5k'
    'YXJkIGRldmlhdGlvbicpKQogICAgcmF3PXRvcmNoLmZyb21fbnVtcHkoaW1hZ2VzLnRyYW5zcG9zZSgwLDMsMSwyKS5jb3B5KCkp'
    'LnRvKCdjdWRhJyk7IHk9dG9yY2guZnJvbV9udW1weShsYWJlbHMpLnRvKCdjdWRhJykKICAgIG11PXRvcmNoLnRlbnNvcihtZWFu'
    'LGR0eXBlPXRvcmNoLmZsb2F0MzIsZGV2aWNlPSdjdWRhJylbTm9uZSw6LE5vbmUsTm9uZV07IHNkPXRvcmNoLnRlbnNvcihzdGQs'
    'ZHR5cGU9dG9yY2guZmxvYXQzMixkZXZpY2U9J2N1ZGEnKVtOb25lLDosTm9uZSxOb25lXQogICAgdHJhaW5fZ3B1PXRvcmNoLnRl'
    'bnNvcih0cmFpbixkZXZpY2U9J2N1ZGEnKTsgZGV2X2dwdT10b3JjaC50ZW5zb3IoZGV2LGRldmljZT0nY3VkYScpCiAgICBzb3Vy'
    'Y2Vfc2hhPWhhc2hsaWIuc2hhMjU2KFBhdGgoX19maWxlX18pLnJlYWRfYnl0ZXMoKSkuaGV4ZGlnZXN0KCk7IHJlY29yZHM9W10K'
    'ICAgIGNvbmZpZz1kaWN0KCoqQ0ZHLGFyY2hpdGVjdHVyZT0ncGF0Y2gyIHN0ZW07IHRocmVlIGNvbnYzIGJsb2NrczsgZm91cnRo'
    'IGJsb2NrIHNpbmdsZSBvciB0d28gZXhwZXJ0IGNvbnYzcyB3aXRoIGlucHV0LWNvbmRpdGlvbmVkIHNjYWxhciBzb2Z0bWF4IG9y'
    'IHplcm8taW5wdXQgc3RhdGljIGdhdGU7IGFsbCBHRUxVL0JOOyBnbG9iYWwgYXZlcmFnZSBhbmQgbGluZWFyMTAwOyBzYW1lIFcg'
    'cGx1cy9taW51cyBpbmRlcGVuZGVudCBub3JtYWwgcGVydHVyYmF0aW9uIGF0IDAuMDEgVyBwb3B1bGF0aW9uIHN0ZDsgZXF1YWwg'
    'Ymlhc2VzOyB6ZXJvIGZpbmFsIGdhdGUgd2VpZ2h0IGFuZCBiaWFzOyBnYXRlIExpbmVhcjY0LTE2L0dFTFUvTGluZWFyMTYtMiwg'
    'dGVtcGVyYXR1cmUgMTsgb3RoZXIgdGVuc29ycyBzaGFyZWQnLHByZWNpc2lvbj0nZGV0ZXJtaW5pc3RpYyBmbG9hdDMyLCBURjMy'
    'IGRpc2FibGVkJyxzZWxlY3Rpb249J21pbmltdW0gb2JzZXJ2ZWQgYXVnbWVudGVkIG1pbmliYXRjaCBDRSBiZWZvcmUgb3B0aW1p'
    'emVyIHN0ZXAgYWZ0ZXIgQk4gdXBkYXRlIGFuZCBmaW5hbCBjYW5vbmljYWwgdHJhaW4gQ0U7IG1peGVkIHNjb3Blcycsc291cmNl'
    'X3NoYTI1Nj1zb3VyY2Vfc2hhKQogICAgZm9yIGlkeCxzZWVkIGluIGVudW1lcmF0ZShDRkdbJ3NlZWRzJ10pOgogICAgICAgIG9y'
    'ZGVyLGNyb3AsZmxpcD1zY2hlZHVsZShzZWVkLENGR1snZXBvY2hzJ10sbGVuKHRyYWluKSkKICAgICAgICBzY2hlZD13cml0ZXIu'
    'c2F2ZV9kYXRhc2V0KGYnc2NoZWR1bGUte3NlZWR9JyxkaWN0KG9yZGVyPW9yZGVyLGNyb3A9Y3JvcCxmbGlwPWZsaXApLHByb3Zl'
    'bmFuY2U9ZGljdChzZWVkPXNlZWQsZ2VuZXJhdG9yPSdQQ0c2NCBzZWVkKzEwMDAwMCtlcG9jaCoxMDAwMDAzOyBwZXJtdXRhdGlv'
    'bixjcm9wLGZsaXAnLG51bXB5X3ZlcnNpb249bnAuX192ZXJzaW9uX18saW5kZXhpbmc9J29yZGVyIGluZGV4ZXMgc29ydGVkIHRy'
    'YWluIElEczsgY3JvcCBhbmQgZmxpcCBpbmRleGVkIGJ5IG9yZGVyIHBvc2l0aW9uJykpCiAgICAgICAgdG9yY2guY3VkYS5tYW51'
    'YWxfc2VlZF9hbGwoc2VlZCkKICAgICAgICBzaW5nbGVfc3RhdGUsZHluYW1pY19zdGF0ZT1wYWlyZWRfaW5pdGlhbF9zdGF0ZXMo'
    'c2VlZCkKICAgICAgICByZWZzPVtzb3VyY2VbJ3BhdGgnXSxkYXRhWydwYXRoJ10sc2NoZWRbJ3BhdGgnXV0KICAgICAgICBhcm1z'
    'PUFSTVNbaWR4JTM6XStBUk1TWzppZHglM10KICAgICAgICBmb3IgYXJtIGluIGFybXM6CiAgICAgICAgICAgIGluaXRpYWw9c2lu'
    'Z2xlX3N0YXRlIGlmIGFybT09J3NpbmdsZScgZWxzZSBkeW5hbWljX3N0YXRlCiAgICAgICAgICAgIG1vZGVsPU1vZGVsKGFybSk7'
    'IG1vZGVsLmxvYWRfc3RhdGVfZGljdChpbml0aWFsLHN0cmljdD1UcnVlKTsgYXNzZXJ0IGFsbCh0b3JjaC5lcXVhbCh2LGluaXRp'
    'YWxba10pIGZvciBrLHYgaW4gbW9kZWwuc3RhdGVfZGljdCgpLml0ZW1zKCkpCiAgICAgICAgICAgIGluaXRpYWxfcmVmPXdyaXRl'
    'ci5zYXZlX2NoZWNrcG9pbnQoZid7c2VlZH0te2FybX0taW5pdGlhbCcsaW5pdGlhbCxhcm09YXJtLHNlZWQ9c2VlZCxzZWxlY3Rp'
    'b249ZGljdChtZXRyaWM9J3BhaXJlZCBpbml0aWFsaXphdGlvbicpLGNvbmZpZz1jb25maWcsZGF0YXNldHM9cmVmcykKICAgICAg'
    'ICAgICAgbW9kZWwudG8oJ2N1ZGEnKTsgYXNzZXJ0IGFsbCh0LmlzX2N1ZGEgZm9yIHQgaW4gbGlzdChtb2RlbC5wYXJhbWV0ZXJz'
    'KCkpK2xpc3QobW9kZWwuYnVmZmVycygpKSkKICAgICAgICAgICAgb3B0aW1pemVyPXRvcmNoLm9wdGltLkFkYW1XKG1vZGVsLnBh'
    'cmFtZXRlcnMoKSxscj1DRkdbJ2xyJ10sd2VpZ2h0X2RlY2F5PUNGR1snd2VpZ2h0X2RlY2F5J10pCiAgICAgICAgICAgIGJlc3Q9'
    'QmVzdENoZWNrcG9pbnQoc2NvcGU9Y29uZmlnWydzZWxlY3Rpb24nXSk7IHRyYWNlcz1bXTsgZXBvY2hfdGltZXM9W107IHN0ZXA9'
    'MAogICAgICAgICAgICB0b3JjaC5jdWRhLnJlc2V0X3BlYWtfbWVtb3J5X3N0YXRzKCk7IHRvcmNoLmN1ZGEuc3luY2hyb25pemUo'
    'KTsgc3RhcnRlZD10aW1lLm1vbm90b25pYygpCiAgICAgICAgICAgIGZvciBlcG9jaCBpbiByYW5nZShDRkdbJ2Vwb2NocyddKToK'
    'ICAgICAgICAgICAgICAgIG1vZGVsLnRyYWluKCk7IHRpY2s9dGltZS5tb25vdG9uaWMoKQogICAgICAgICAgICAgICAgZm9yIHN0'
    'YXJ0IGluIHJhbmdlKDAsbGVuKHRyYWluKSxDRkdbJ2JhdGNoJ10pOgogICAgICAgICAgICAgICAgICAgIHN0b3A9bWluKHN0YXJ0'
    'K0NGR1snYmF0Y2gnXSxsZW4odHJhaW4pKTsgaXg9dHJhaW5fZ3B1W3RvcmNoLmFzX3RlbnNvcihvcmRlcltlcG9jaCxzdGFydDpz'
    'dG9wXS5hc3R5cGUobnAuaW50NjQpLGRldmljZT0nY3VkYScpXQogICAgICAgICAgICAgICAgICAgIGM9dG9yY2guYXNfdGVuc29y'
    'KGNyb3BbZXBvY2gsc3RhcnQ6c3RvcF0uYXN0eXBlKG5wLmludDY0KSxkZXZpY2U9J2N1ZGEnKTsgZj10b3JjaC5hc190ZW5zb3Io'
    'ZmxpcFtlcG9jaCxzdGFydDpzdG9wXSxkZXZpY2U9J2N1ZGEnKQogICAgICAgICAgICAgICAgICAgIG9wdGltaXplci56ZXJvX2dy'
    'YWQoc2V0X3RvX25vbmU9VHJ1ZSk7IGxvc3M9Ri5jcm9zc19lbnRyb3B5KG1vZGVsKGlucHV0cyhyYXcsaXgsbXUsc2QsYyxmKSks'
    'eVtpeF0pOyB2YWx1ZT1mbG9hdChsb3NzLmRldGFjaCgpKQogICAgICAgICAgICAgICAgICAgIGlmIG5vdCBtYXRoLmlzZmluaXRl'
    'KHZhbHVlKTogcmFpc2UgRmxvYXRpbmdQb2ludEVycm9yKCdub25maW5pdGUgbG9zcycpCiAgICAgICAgICAgICAgICAgICAgYmVz'
    'dC5vYnNlcnZlKG1vZGVsLHZhbHVlLHN0ZXApOyB0cmFjZXMuYXBwZW5kKFtzdGVwLGVwb2NoLHN0YXJ0LHN0b3Atc3RhcnQsdmFs'
    'dWUsMF0pOyBsb3NzLmJhY2t3YXJkKCkKICAgICAgICAgICAgICAgICAgICBmb3IgcCBpbiBtb2RlbC5wYXJhbWV0ZXJzKCk6CiAg'
    'ICAgICAgICAgICAgICAgICAgICAgIGlmIHAuZ3JhZCBpcyBub3QgTm9uZTogdG9yY2guX2Fzc2VydF9hc3luYyh0b3JjaC5pc2Zp'
    'bml0ZShwLmdyYWQpLmFsbCgpLCdub25maW5pdGUgZ3JhZGllbnQnKQogICAgICAgICAgICAgICAgICAgIG9wdGltaXplci5zdGVw'
    'KCk7IHN0ZXArPTEKICAgICAgICAgICAgICAgIHRvcmNoLmN1ZGEuc3luY2hyb25pemUoKTsgZXBvY2hfdGltZXMuYXBwZW5kKHRp'
    'bWUubW9ub3RvbmljKCktdGljayk7IGV2ZW50KCdlcG9jaF9jb21wbGV0ZScsc2VlZD1zZWVkLGFybT1hcm0sZXBvY2g9ZXBvY2gr'
    'MSx1cGRhdGVzPXN0ZXAsc2Vjb25kcz1lcG9jaF90aW1lc1stMV0pCiAgICAgICAgICAgIHRvcmNoLmN1ZGEuc3luY2hyb25pemUo'
    'KTsgdHJhaW5pbmdfc2Vjb25kcz10aW1lLm1vbm90b25pYygpLXN0YXJ0ZWQKICAgICAgICAgICAgaWYgbm90IGFsbCh0b3JjaC5p'
    'c2Zpbml0ZSh2KS5hbGwoKS5pdGVtKCkgZm9yIHYgaW4gbW9kZWwuc3RhdGVfZGljdCgpLnZhbHVlcygpKTogcmFpc2UgRmxvYXRp'
    'bmdQb2ludEVycm9yKCdub25maW5pdGUgc3RhdGUnKQogICAgICAgICAgICBmaW5hbF90cmFpbixfPWV2YWx1YXRlKG1vZGVsLHJh'
    'dyx5LHRyYWluX2dwdSxtdSxzZCk7IGJlc3Qub2JzZXJ2ZShtb2RlbCxmaW5hbF90cmFpblsnY2UnXSxzdGVwKTsgdHJhY2VzLmFw'
    'cGVuZChbc3RlcCxDRkdbJ2Vwb2NocyddLC0xLGxlbih0cmFpbiksZmluYWxfdHJhaW5bJ2NlJ10sMV0pCiAgICAgICAgICAgIGZp'
    'bmFsX2RldixkZXZfbG9naXRzPWV2YWx1YXRlKG1vZGVsLHJhdyx5LGRldl9ncHUsbXUsc2QpCiAgICAgICAgICAgIGlmIGRldl9s'
    'b2dpdHMuc2hhcGUhPShsZW4oZGV2KSwxMDApIG9yIG5vdCBucC5pc2Zpbml0ZShkZXZfbG9naXRzKS5hbGwoKSBvciBub3QgYWxs'
    'KG1hdGguaXNmaW5pdGUodikgZm9yIHYgaW4gKCpmaW5hbF90cmFpbi52YWx1ZXMoKSwqZmluYWxfZGV2LnZhbHVlcygpKSk6IHJh'
    'aXNlIEZsb2F0aW5nUG9pbnRFcnJvcignaW52YWxpZCBmaW5hbCBlbmRwb2ludCcpCiAgICAgICAgICAgIHJvdXRpbmc9cm91dGlu'
    'Z19kaWFnbm9zdGljcyhtb2RlbCxyYXcsZGV2X2dwdSxtdSxzZCkKICAgICAgICAgICAgd3JpdGVyLnNhdmVfZGF0YXNldChmJ3tz'
    'ZWVkfS17YXJtfS1kZXYtbG9naXRzJyxkaWN0KGlkcz1kZXYsbGFiZWxzPWxhYmVsc1tkZXZdLGxvZ2l0cz1kZXZfbG9naXRzKSxw'
    'cm92ZW5hbmNlPWRpY3QoY2hlY2twb2ludD0nZml4ZWQgZmluYWwgc3RhdGUnLG9yZGVyZWQ9J3NvcnRlZCBkZXZlbG9wbWVudCBJ'
    'RHMnLHNwbGl0PWRhdGFbJ3BhdGgnXSkpCiAgICAgICAgICAgIGZpbmFsPXdyaXRlci5zYXZlX2NoZWNrcG9pbnQoZid7c2VlZH0t'
    'e2FybX0tZmluYWwnLG1vZGVsLGFybT1hcm0sc2VlZD1zZWVkLHNlbGVjdGlvbj1kaWN0KG1ldHJpYz0nZml4ZWQgZmluYWwgdXBk'
    'YXRlJyxzdGVwPXN0ZXApLGNvbmZpZz1jb25maWcsZGF0YXNldHM9cmVmcykKICAgICAgICAgICAgYXNzZXJ0IGJlc3Qub2JzZXJ2'
    'YXRpb25zPT1sZW4odHJhY2VzKT09c3RlcCsxIGFuZCBiZXN0LnZhbHVlPT1taW4oeFs0XSBmb3IgeCBpbiB0cmFjZXMpCiAgICAg'
    'ICAgICAgIHNlbGVjdGVkPWJlc3Quc2F2ZSh3cml0ZXIsZid7c2VlZH0te2FybX0tdHJhaW5pbmctc2VsZWN0ZWQnLGFybT1hcm0s'
    'c2VlZD1zZWVkLGNvbmZpZz1jb25maWcsZGF0YXNldHM9cmVmcykKICAgICAgICAgICAgd3JpdGVyLnNhdmVfZGF0YXNldChmJ3tz'
    'ZWVkfS17YXJtfS10cmFjZScsZGljdChvYnNlcnZhdGlvbnM9bnAuYXNhcnJheSh0cmFjZXMsZHR5cGU9bnAuZmxvYXQ2NCkpLHBy'
    'b3ZlbmFuY2U9ZGljdChjb2x1bW5zPVsnc3RlcCcsJ2Vwb2NoJywnb3JkZXJfc3RhcnQnLCdjb3VudCcsJ2NlJywnc2NvcGVfMF9h'
    'dWdtZW50ZWRfbWluaWJhdGNoXzFfZmluYWxfY2Fub25pY2FsX3RyYWluaW5nJ10pKQogICAgICAgICAgICByZWNvcmQ9ZGljdChz'
    'ZWVkPXNlZWQsYXJtPWFybSxhcm1fb3JkZXI9bGlzdChhcm1zKSxpbml0aWFsPWluaXRpYWxfcmVmWydwYXRoJ10sZmluYWw9Zmlu'
    'YWxbJ3BhdGgnXSxzZWxlY3RlZD1zZWxlY3RlZFsncGF0aCddLHVwZGF0ZXM9c3RlcCx0cmFpbmluZ19zZWNvbmRzPXRyYWluaW5n'
    'X3NlY29uZHMsZXBvY2hfc2Vjb25kcz1lcG9jaF90aW1lcyxwZWFrX2FsbG9jYXRlZF9ieXRlcz10b3JjaC5jdWRhLm1heF9tZW1v'
    'cnlfYWxsb2NhdGVkKCkscGVha19yZXNlcnZlZF9ieXRlcz10b3JjaC5jdWRhLm1heF9tZW1vcnlfcmVzZXJ2ZWQoKSxwYXJhbWV0'
    'ZXJzPXN1bShwLm51bWVsKCkgZm9yIHAgaW4gbW9kZWwucGFyYW1ldGVycygpKSxmaW5hbF90cmFpbj1maW5hbF90cmFpbixmaW5h'
    'bF9kZXY9ZmluYWxfZGV2LHJvdXRpbmc9cm91dGluZykKICAgICAgICAgICAgcmVjb3Jkcy5hcHBlbmQocmVjb3JkKTsgZXZlbnQo'
    'J2ZpdF9jb21wbGV0ZScsKipyZWNvcmQpOyBkZWwgbW9kZWwsb3B0aW1pemVyLGJlc3QKICAgIGlmIGxlbihyZWNvcmRzKSE9bGVu'
    'KENGR1snc2VlZHMnXSkqbGVuKEFSTVMpIG9yIHsoclsnc2VlZCddLHJbJ2FybSddKSBmb3IgciBpbiByZWNvcmRzfSE9eyhzZWVk'
    'LGFybSkgZm9yIHNlZWQgaW4gQ0ZHWydzZWVkcyddIGZvciBhcm0gaW4gQVJNU306IHJhaXNlIFZhbHVlRXJyb3IoJ2luY29tcGxl'
    'dGUgb3IgZHVwbGljYXRlIGZpdHMnKQogICAgaWYgYW55KHJbJ3VwZGF0ZXMnXSE9Q0ZHWydlcG9jaHMnXSptYXRoLmNlaWwobGVu'
    'KHRyYWluKS9DRkdbJ2JhdGNoJ10pIGZvciByIGluIHJlY29yZHMpOiByYWlzZSBWYWx1ZUVycm9yKCdpbmNvcnJlY3QgdXBkYXRl'
    'IGNvdW50JykKICAgIHBhaXJlZD1bbmV4dChyWydmaW5hbF9kZXYnXVsnY2UnXSBmb3IgciBpbiByZWNvcmRzIGlmIHJbJ3NlZWQn'
    'XT09cyBhbmQgclsnYXJtJ109PSdkeW5hbWljJyktbmV4dChyWydmaW5hbF9kZXYnXVsnY2UnXSBmb3IgciBpbiByZWNvcmRzIGlm'
    'IHJbJ3NlZWQnXT09cyBhbmQgclsnYXJtJ109PSdzdGF0aWMnKSBmb3IgcyBpbiBDRkdbJ3NlZWRzJ11dCiAgICByZWZlcmVuY2U9'
    'W25leHQoclsnZmluYWxfZGV2J11bJ2NlJ10gZm9yIHIgaW4gcmVjb3JkcyBpZiByWydzZWVkJ109PXMgYW5kIHJbJ2FybSddPT0n'
    'ZHluYW1pYycpLW5leHQoclsnZmluYWxfZGV2J11bJ2NlJ10gZm9yIHIgaW4gcmVjb3JkcyBpZiByWydzZWVkJ109PXMgYW5kIHJb'
    'J2FybSddPT0nc2luZ2xlJykgZm9yIHMgaW4gQ0ZHWydzZWVkcyddXQogICAgZGV0YWlscz1kaWN0KGNvbmZpZz1jb25maWcscmVj'
    'b3Jkcz1yZWNvcmRzLHBhaXJlZF9keW5hbWljX21pbnVzX3N0YXRpY19jZT1wYWlyZWRfc3VtbWFyeShwYWlyZWQpLHBhaXJlZF9k'
    'eW5hbWljX21pbnVzX3NpbmdsZV9jZT1wYWlyZWRfc3VtbWFyeShyZWZlcmVuY2UpLGFydGlmYWN0X2hlbHBlcl9zaGEyNTY9aGFz'
    'aGxpYi5zaGEyNTYoUGF0aChhcnRpZmFjdHMuX19maWxlX18pLnJlYWRfYnl0ZXMoKSkuaGV4ZGlnZXN0KCkscHl0aG9uX3ZlcnNp'
    'b249c3lzLnZlcnNpb24scGxhdGZvcm09cGxhdGZvcm0ucGxhdGZvcm0oKSxudW1weV92ZXJzaW9uPW5wLl9fdmVyc2lvbl9fLGRh'
    'dGFzZXRzX3ZlcnNpb249ZGF0YXNldHMuX192ZXJzaW9uX18saHVnZ2luZ2ZhY2VfaHViX3ZlcnNpb249aHVnZ2luZ2ZhY2VfaHVi'
    'Ll9fdmVyc2lvbl9fLGN1ZG5uX3ZlcnNpb249dG9yY2guYmFja2VuZHMuY3Vkbm4udmVyc2lvbigpLGNyaXRlcmlvbl9tZXQ9c3Vt'
    'KHBhaXJlZCkvbGVuKHBhaXJlZCk8PUNGR1sndGhyZXNob2xkJ10gYW5kIGFsbCh2PDAgZm9yIHYgaW4gcGFpcmVkKSxkZXZpY2U9'
    'dG9yY2guY3VkYS5nZXRfZGV2aWNlX25hbWUoKSx0b3JjaF92ZXJzaW9uPXRvcmNoLl9fdmVyc2lvbl9fLGN1ZGFfdmVyc2lvbj10'
    'b3JjaC52ZXJzaW9uLmN1ZGEsZWxhcHNlZF9zZWNvbmRzPXRpbWUubW9ub3RvbmljKCktU1RBUlQpCiAgICAoT1VULydkZXRhaWxz'
    'Lmpzb24nKS53cml0ZV90ZXh0KGpzb24uZHVtcHMoZGV0YWlscyxpbmRlbnQ9MixhbGxvd19uYW49RmFsc2UpKQogICAgKE9VVC8n'
    'cmVzdWx0Lmpzb24nKS53cml0ZV90ZXh0KGpzb24uZHVtcHMoZGljdChtZXRyaWNzPWRpY3QoY3JpdGVyaW9uX21ldD1pbnQoZGV0'
    'YWlsc1snY3JpdGVyaW9uX21ldCddKSx0cmFpbmluZ19zZWNvbmRzPXN1bShyWyd0cmFpbmluZ19zZWNvbmRzJ10gZm9yIHIgaW4g'
    'cmVjb3JkcyksZml0cz1sZW4ocmVjb3JkcykscGVha19jdWRhX21lbW9yeV9ieXRlcz1tYXgoclsncGVha19hbGxvY2F0ZWRfYnl0'
    'ZXMnXSBmb3IgciBpbiByZWNvcmRzKSksc3VtbWFyeT0ncHJvcG9zZWQgY29uZGl0aW9uYWwtY29udm9sdXRpb24gcGFpcmVkIHNj'
    'cmVlbjsgaW5kZXBlbmRlbnRseSBhdWRpdCBiZWZvcmUgaW50ZXJwcmV0YXRpb24nKSxpbmRlbnQ9MixhbGxvd19uYW49RmFsc2Up'
    'KQoKaWYgX19uYW1lX189PSdfX21haW5fXyc6CiAgICB0cnk6IG1haW4oKQogICAgZXhjZXB0IEV4Y2VwdGlvbjoKICAgICAgICBP'
    'VVQubWtkaXIocGFyZW50cz1UcnVlLGV4aXN0X29rPVRydWUpOyAoT1VULydmYWlsdXJlLXRyYWNlYmFjay50eHQnKS53cml0ZV90'
    'ZXh0KHRyYWNlYmFjay5mb3JtYXRfZXhjKCkpOyByYWlzZQo='
)
SOURCE_SHA = '03883f110960c4bfb857a37b572411bd7ef536ab6115ff52354f70eeb0df697e'
ROOT = Path('/output')
OUT = ROOT / 'calibration'
output_owned = False
START = time.monotonic()
def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()
def size_tree(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
def cgroup_memory():
    values = {}
    for name in ('memory.current', 'memory.peak', 'memory.max'):
        try: values[name] = Path('/sys/fs/cgroup', name).read_text().strip()
        except OSError: values[name] = None
    return values
def sync(): torch.cuda.synchronize()
def timed(fn):
    sync(); t = time.monotonic(); result = fn(); sync()
    return result, time.monotonic() - t

def main():
    global output_owned
    source_bytes = base64.b64decode(EMBEDDED_SOURCE, validate=True)
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_SHA: raise RuntimeError('science source hash mismatch')
    sci = types.ModuleType('science_candidate'); sci.__file__ = 'embedded-science.py'
    exec(compile(source_bytes, sci.__file__, 'exec'), sci.__dict__)
    os.environ['HF_HOME'] = str(OUT / 'hf-cache')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'result.json').exists(): raise RuntimeError('result already exists')
    OUT.mkdir(exist_ok=False)
    output_owned = True
    writer = ArtifactWriter(OUT)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    free_before = shutil.disk_usage(OUT).free
    prov = OUT / 'provenance'; prov.mkdir()
    refs = []
    (prov / 'science.py').write_bytes(source_bytes)
    for path, name in ((Path(__file__), 'calibration.py'), (prov / 'science.py', 'science.py'), (Path(artifacts.__file__), 'artifacts.py')):
        target = prov / name
        if path != target: shutil.copyfile(path, target)
        refs.append(writer.register_file('datasets', target, name, format='python', source={'sha256': digest(target)})['path'])
    from huggingface_hub import HfApi, hf_hub_download
    import huggingface_hub, datasets
    info = HfApi(token=False).dataset_info('uoft-cs/cifar100', revision=sci.REV, timeout=20)
    if info.sha != sci.REV: raise ValueError('revision mismatch')
    path = Path(hf_hub_download(repo_id='uoft-cs/cifar100', repo_type='dataset', filename=sci.FILE, revision=sci.REV, token=False, local_dir=str(OUT/'hf-source')))
    if digest(path) != sci.SHA: raise ValueError('source hash mismatch')
    source = writer.register_file('datasets', path, 'official-train-source', format='parquet', source={'repository': 'uoft-cs/cifar100', 'split': 'train', 'filename': sci.FILE, 'url': f'https://huggingface.co/datasets/uoft-cs/cifar100/blob/{sci.REV}/{sci.FILE}', 'revision': sci.REV, 'sha256': sci.SHA, 'license_from_dataset_card': info.card_data.get('license') if info.card_data else None, 'preprocessing': 'decode img as RGB uint8 and fine_label as int64; stratified train split and train-only channel mean/std'})
    ds = datasets.load_dataset('parquet', data_files={'train': str(path)}, split='train', cache_dir=str(OUT/'hf-cache'/'decoded'))
    images = np.stack([np.asarray(row['img'].convert('RGB'), dtype=np.uint8) for row in ds]); labels = np.asarray(ds['fine_label'], dtype=np.int64)
    if images.shape != (50000,32,32,3) or not np.array_equal(np.bincount(labels,minlength=100),np.full(100,500)): raise ValueError('unexpected data')
    rng = np.random.default_rng(sci.CFG['split_seed']); train=[]; dev=[]
    for c in range(100):
        ids=rng.permutation(np.flatnonzero(labels==c)); train.extend(ids[:450]); dev.extend(ids[450:])
    train=np.asarray(sorted(train)); dev=np.asarray(sorted(dev))
    if len(train)!=45000 or len(dev)!=5000 or len(np.unique(np.concatenate([train,dev])))!=50000 or np.intersect1d(train,dev).size or not np.array_equal(np.bincount(labels[train],minlength=100),np.full(100,450)) or not np.array_equal(np.bincount(labels[dev],minlength=100),np.full(100,50)): raise ValueError('invalid split')
    decoded_sha=hashlib.sha256(images.tobytes()+labels.tobytes()).hexdigest()
    pixels=images[train].astype(np.float64)/255; mean=pixels.mean((0,1,2)); std=pixels.std((0,1,2)); del pixels
    data=writer.save_dataset('train-split',dict(images=images,labels=labels,train_ids=train,dev_ids=dev,mean=mean,std=std),provenance={'source':source,'decoded_sha256_images_then_labels':decoded_sha,'split_seed':sci.CFG['split_seed'],'official_test':'never requested'})
    raw=torch.from_numpy(images.transpose(0,3,1,2).copy()).to('cuda'); y=torch.from_numpy(labels).to('cuda')
    mu=torch.tensor(mean,dtype=torch.float32,device='cuda')[None,:,None,None]; sd=torch.tensor(std,dtype=torch.float32,device='cuda')[None,:,None,None]
    train_gpu=torch.tensor(train,device='cuda'); seed=4801; epochs=2
    order,crop,flip=sci.schedule(seed,epochs,len(train))
    sched=writer.save_dataset('schedule-4801',dict(order=order,crop=crop,flip=flip),provenance={'generator':'science schedule; position-indexed crop and flip','numpy':np.__version__})
    refs += [source['path'],data['path'],sched['path']]
    setup_seconds=time.monotonic()-START
    torch.cuda.manual_seed_all(seed); single_state,dynamic_state=sci.paired_initial_states(seed)
    records=[]; config={'seed':seed,'epochs':epochs,'batch':256,'lr':.001,'weight_decay':.01,'selection':'minimum observed augmented minibatch CE before optimizer step after BN update and final canonical train CE; mixed scopes','reload_rtol':1e-5,'reload_atol':1e-6}
    for arm in sci.ARMS:
        initial=single_state if arm=='single' else dynamic_state
        model=sci.Model(arm); model.load_state_dict(initial,strict=True)
        initial_ref=writer.save_checkpoint(f'{seed}-{arm}-initial',initial,arm=arm,seed=seed,selection={'metric':'paired initialization'},config=config,datasets=refs)
        model.to('cuda'); optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        best=BestCheckpoint(scope=config['selection']); traces=[]; step=0; presentations=0; epoch_times=[]; forward=backward=optim=0.
        sync(); baseline_allocated=torch.cuda.memory_allocated(); baseline_reserved=torch.cuda.memory_reserved(); torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
        for epoch in range(epochs):
            model.train(); sync(); tick=time.monotonic()
            for start in range(0,len(train),256):
                stop=min(start+256,len(train)); ix=train_gpu[torch.as_tensor(order[epoch,start:stop].astype(np.int64),device='cuda')]
                c=torch.as_tensor(crop[epoch,start:stop].astype(np.int64),device='cuda'); f=torch.as_tensor(flip[epoch,start:stop],device='cuda')
                optimizer.zero_grad(set_to_none=True)
                loss,dt=timed(lambda: F.cross_entropy(model(sci.inputs(raw,ix,mu,sd,c,f)),y[ix])); forward+=dt
                value=float(loss.detach())
                if not math.isfinite(value): raise FloatingPointError('nonfinite loss')
                best.observe(model,value,step); traces.append([step,epoch,start,stop-start,value,0])
                _,dt=timed(loss.backward); backward+=dt
                for p in model.parameters():
                    if p.grad is not None: torch._assert_async(torch.isfinite(p.grad).all(),'nonfinite gradient')
                _,dt=timed(optimizer.step); optim+=dt
                step+=1; presentations+=stop-start
            sync(); epoch_times.append(time.monotonic()-tick)
        training_seconds=time.monotonic()-started
        if step!=352 or presentations!=90000 or traces[-1][3]!=200: raise ValueError('invalid counts or partial batch')
        if not all(torch.isfinite(v).all().item() for v in model.state_dict().values()): raise FloatingPointError('nonfinite state')
        (train_metric,_),inference=timed(lambda: sci.evaluate(model,raw,y,train_gpu,mu,sd))
        if not all(math.isfinite(v) for v in train_metric.values()): raise FloatingPointError('nonfinite train inference')
        best.observe(model,train_metric['ce'],step); traces.append([step,epochs,-1,len(train),train_metric['ce'],1])
        assert best.observations==len(traces)==step+1 and best.value==min(x[4] for x in traces)
        model.eval(); fixed=train_gpu[:256]
        with torch.no_grad(): expected=model(sci.inputs(raw,fixed,mu,sd)).detach().cpu()
        tick=time.monotonic()
        final=writer.save_checkpoint(f'{seed}-{arm}-final',model,arm=arm,seed=seed,selection={'metric':'fixed final update','step':step},config=config,datasets=refs)
        selected=best.save(writer,f'{seed}-{arm}-training-selected',arm=arm,seed=seed,config=config,datasets=refs)
        trace=writer.save_dataset(f'{seed}-{arm}-trace',dict(observations=np.asarray(traces,dtype=np.float64)),provenance={'columns':['step','epoch','order_start','count','ce','scope_0_augmented_minibatch_1_final_canonical_training']})
        serialization=time.monotonic()-tick
        restored=sci.Model(arm); restored.load_state_dict(load_json_state(OUT/final['path']),strict=True); restored.to('cuda').eval()
        with torch.no_grad(): actual=restored(sci.inputs(raw,fixed,mu,sd)).cpu()
        if not torch.isfinite(actual).all() or not torch.allclose(expected,actual,rtol=1e-5,atol=1e-6): raise ValueError('final checkpoint replay mismatch')
        selected_state=load_json_state(OUT/selected['path'])
        if selected_state.keys()!=best.state.keys() or any(not torch.equal(selected_state[k],best.state[k]) for k in best.state): raise ValueError('selected checkpoint state mismatch')
        selected_model=sci.Model(arm); selected_model.load_state_dict(best.state,strict=True); selected_model.to('cuda').eval()
        restored.load_state_dict(selected_state,strict=True); restored.eval()
        with torch.no_grad():
            selected_expected=selected_model(sci.inputs(raw,fixed,mu,sd)).cpu()
            selected_actual=restored(sci.inputs(raw,fixed,mu,sd)).cpu()
        if not torch.isfinite(selected_actual).all() or not torch.allclose(selected_expected,selected_actual,rtol=1e-5,atol=1e-6): raise ValueError('selected checkpoint replay mismatch')
        if best.step != next(row[0] for row in traces if row[4]==best.value): raise ValueError('selection is not earliest minimum')
        records.append(dict(arm=arm,initial=initial_ref['path'],final=final['path'],selected=selected['path'],trace=trace['path'],updates=step,presentations=presentations,partial_batch=200,epoch_seconds=epoch_times,training_seconds=training_seconds,forward_seconds=forward,backward_seconds=backward,optimizer_seconds=optim,train_inference_seconds=inference,serialization_seconds=serialization,replay_max_abs=float((expected-actual).abs().max()),selected_replay_max_abs=float((selected_expected-selected_actual).abs().max()),selected_replay_scope='eval-mode fixed training inputs; does not reproduce training-mode selection loss',baseline_allocated_bytes=baseline_allocated,baseline_reserved_bytes=baseline_reserved,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        print(json.dumps({'event':'fit_complete','record':records[-1]},allow_nan=False),flush=True)
        del model,restored,selected_model,optimizer,best
    try: driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).strip()
    except Exception as e: driver='unavailable: '+repr(e)
    result={'config':config,'arm_order':list(sci.ARMS),'records':records,'setup_seconds':setup_seconds,'elapsed_seconds':time.monotonic()-START,'source_sha256':SOURCE_SHA,'helper_sha256':digest(Path(artifacts.__file__)),'calibration_sha256':digest(Path(__file__)),'gpu':torch.cuda.get_device_name(),'driver':driver,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'torch':torch.__version__,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'datasets':datasets.__version__,'huggingface_hub':huggingface_hub.__version__,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_benchmark':torch.backends.cudnn.benchmark,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'process_peak_rss_kib_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'output_bytes_before_summary':size_tree(OUT),'cache_bytes':size_tree(OUT/'hf-cache'),'disk_free_before':free_before,'disk_free_after_before_summary':shutil.disk_usage(OUT).free,'timing_note':'synchronization every phase adds overhead; forward includes preprocessing and loss; serialization includes manifest writes'}
    result['cgroup_v2_memory_before_summary'] = cgroup_memory()
    result['accounting_scope'] = 'elapsed and disk sampled before summary, integrity and result publication; disk is not high-water; cgroup peak may include prior processes; GPU peaks include replay'
    (OUT/'calibration.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    result_bytes=json.dumps({'metrics': {'arms_completed': len(records), 'updates_per_arm': step, 'presentations_per_arm': presentations}, 'summary': 'Real-data calibration completed; no development or test endpoint was evaluated. Details are in calibration/calibration.json.'}, allow_nan=False).encode()
    files={p.relative_to(OUT).as_posix():{'sha256':digest(p),'bytes':p.stat().st_size} for p in OUT.rglob('*') if p.is_file() and p.name!='integrity.json'}
    files['../result.json']={'sha256':hashlib.sha256(result_bytes).hexdigest(),'bytes':len(result_bytes)}
    (OUT/'integrity.json').write_text(json.dumps(files,indent=2,allow_nan=False))
    # Publish the completed result atomically and refuse to replace any existing file.
    result_tmp=OUT/'.result.json.tmp'
    with result_tmp.open('xb') as f: f.write(result_bytes)
    os.link(result_tmp, ROOT/'result.json')
    result_tmp.unlink()
if __name__=='__main__':
    try: main()
    except Exception:
        if output_owned:
            try:
                with (OUT/'calibration-failure-traceback.txt').open('x') as f: f.write(traceback.format_exc())
            except OSError as error:
                print('Could not preserve failure traceback:', error, file=sys.stderr)
        raise
