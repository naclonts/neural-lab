"""Synthetic-only diagnostic of frozen anti-alias implementation; no training or data access."""
import hashlib
import base64
import types
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from neural_lab.artifacts import ArtifactWriter, BestCheckpoint, load_json_state

EXPECTED = '517c185d6bd45d3f889316b5ee170d7258142733e4ecf8591f48ae56a17e0bf1'
EMBEDDED_SOURCE = (
    'IiIiUHJvc3BlY3RpdmUgQ0lGQVItMTAgcHJlLXN0ZW0gYmx1ciBzY3JlZW4uIENVREEgcmVxdWlyZWQ7IHBpbm5lZCBwdWJsaWMg'
    'ZGF0YS4iIiIKaW1wb3J0IG9zCm9zLmVudmlyb25bJ0NVQkxBU19XT1JLU1BBQ0VfQ09ORklHJ10gPSAnOjQwOTY6OCcKb3MuZW52'
    'aXJvblsnSEZfSE9NRSddID0gJy9vdXRwdXQvaGYtY2FjaGUnCm9zLmVudmlyb25bJ0hGX0hVQl9ESVNBQkxFX0lNUExJQ0lUX1RP'
    'S0VOJ10gPSAnMScKaW1wb3J0IGNvcHkKaW1wb3J0IGhhc2hsaWIKaW1wb3J0IGpzb24KaW1wb3J0IG1hdGgKaW1wb3J0IHNodXRp'
    'bAppbXBvcnQgdGltZQppbXBvcnQgdHJhY2ViYWNrCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAppbXBvcnQgbnVtcHkgYXMgbnAK'
    'aW1wb3J0IHRvcmNoCmZyb20gdG9yY2ggaW1wb3J0IG5uCmZyb20gdG9yY2gubm4gaW1wb3J0IGZ1bmN0aW9uYWwgYXMgRgpmcm9t'
    'IG5ldXJhbF9sYWIuYXJ0aWZhY3RzIGltcG9ydCBBcnRpZmFjdFdyaXRlciwgQmVzdENoZWNrcG9pbnQKCk9VVCA9IFBhdGgoJy9v'
    'dXRwdXQnKQpTRUVEUyA9ICgzODAxLCAzODAyLCAzODAzLCAzODA0LCAzODA1LCAzODA2KQpBUk1TID0gKCdwbGFpbicsICdiaW5v'
    'bWlhbCcpCkVQT0NIUywgQkFUQ0gsIFdJRFRILCBERVBUSCA9IDEwMCwgMjU2LCA2NCwgNApTT1VSQ0VfU0hBID0gJzg0MjhiNTNh'
    'ODhhMTFhYzM3NDExMTAwNjcwOGRmNTE0NjllMzE1YTIyYWM2ZDY2NDcwYWZkOWM3OGQyYWU4ODMnClNUQVJUID0gdGltZS5tb25v'
    'dG9uaWMoKQoKCmRlZiBzaGEocGF0aCk6CiAgICByZXR1cm4gaGFzaGxpYi5zaGEyNTYoUGF0aChwYXRoKS5yZWFkX2J5dGVzKCkp'
    'LmhleGRpZ2VzdCgpCgoKZGVmIGV2ZW50KGtpbmQsICoqZGF0YSk6CiAgICByb3cgPSBkaWN0KGV2ZW50PWtpbmQsIGVsYXBzZWRf'
    'c2Vjb25kcz10aW1lLm1vbm90b25pYygpLVNUQVJULCAqKmRhdGEpCiAgICB3aXRoIChPVVQvJ3Byb2dyZXNzLmpzb25sJykub3Bl'
    'bignYScpIGFzIGY6CiAgICAgICAgZi53cml0ZShqc29uLmR1bXBzKHJvdywgYWxsb3dfbmFuPUZhbHNlKSsnXG4nKQogICAgcHJp'
    'bnQoanNvbi5kdW1wcyhyb3cpLCBmbHVzaD1UcnVlKQoKCmNsYXNzIE1vZGVsKG5uLk1vZHVsZSk6CiAgICBkZWYgX19pbml0X18o'
    'c2VsZiwgYXJtKToKICAgICAgICBzdXBlcigpLl9faW5pdF9fKCkKICAgICAgICBpZiBhcm0gbm90IGluIEFSTVM6CiAgICAgICAg'
    'ICAgIHJhaXNlIFZhbHVlRXJyb3IoYXJtKQogICAgICAgIHNlbGYuYXJtID0gYXJtCiAgICAgICAgaWYgYXJtID09ICdiaW5vbWlh'
    'bCc6CiAgICAgICAgICAgIHYgPSB0b3JjaC50ZW5zb3IoWzEuLCAyLiwgMS5dKQogICAgICAgICAgICBzZWxmLnJlZ2lzdGVyX2J1'
    'ZmZlcigna2VybmVsJywgKHZbOiwgTm9uZV0qdltOb25lLCA6XS8xNikudmlldygxLCAxLCAzLCAzKS5yZXBlYXQoMywgMSwgMSwg'
    'MSkpCiAgICAgICAgc2VsZi5zdGVtID0gbm4uU2VxdWVudGlhbChubi5Db252MmQoMywgV0lEVEgsIDIsIHN0cmlkZT0yKSwgbm4u'
    'R0VMVSgpLCBubi5CYXRjaE5vcm0yZChXSURUSCkpCiAgICAgICAgc2VsZi5ibG9ja3MgPSBubi5TZXF1ZW50aWFsKCpbbm4uU2Vx'
    'dWVudGlhbChubi5Db252MmQoV0lEVEgsIFdJRFRILCAzLCBwYWRkaW5nPTEpLCBubi5HRUxVKCksIG5uLkJhdGNoTm9ybTJkKFdJ'
    'RFRIKSkgZm9yIF8gaW4gcmFuZ2UoREVQVEgpXSkKICAgICAgICBzZWxmLmhlYWQgPSBubi5MaW5lYXIoV0lEVEgsIDEwKQoKICAg'
    'IGRlZiBmb3J3YXJkKHNlbGYsIHgpOgogICAgICAgIGlmIHNlbGYuYXJtID09ICdiaW5vbWlhbCc6CiAgICAgICAgICAgIHggPSBG'
    'LmNvbnYyZChGLnBhZCh4LCAoMSwgMSwgMSwgMSksIG1vZGU9J3JlZmxlY3QnKSwgc2VsZi5rZXJuZWwsIGdyb3Vwcz0zKQogICAg'
    'ICAgIHJldHVybiBzZWxmLmhlYWQoc2VsZi5ibG9ja3Moc2VsZi5zdGVtKHgpKS5tZWFuKCgyLCAzKSkpCgoKZGVmIHNjaGVkdWxl'
    'KHNlZWQsIG4pOgogICAgb3JkZXIsIGNyb3AsIGZsaXAgPSBbXSwgW10sIFtdCiAgICBmb3IgZXBvY2ggaW4gcmFuZ2UoRVBPQ0hT'
    'KToKICAgICAgICBybmcgPSBucC5yYW5kb20uZGVmYXVsdF9ybmcoc2VlZCsxMDAwMDArZXBvY2gqMTAwMDAwMykKICAgICAgICBv'
    'cmRlci5hcHBlbmQocm5nLnBlcm11dGF0aW9uKG4pLmFzdHlwZShucC5pbnQzMikpCiAgICAgICAgY3JvcC5hcHBlbmQocm5nLmlu'
    'dGVnZXJzKDAsIDksIChuLCAyKSwgZHR5cGU9bnAuaW50MTYpKQogICAgICAgIGZsaXAuYXBwZW5kKHJuZy5pbnRlZ2VycygwLCAy'
    'LCBuLCBkdHlwZT1ucC51aW50OCkpCiAgICByZXR1cm4gbnAuc3RhY2sob3JkZXIpLCBucC5zdGFjayhjcm9wKSwgbnAuc3RhY2so'
    'ZmxpcCkKCgpkZWYgaW5wdXRzKHJhdywgaWRzLCBtZWFuLCBzdGQsIGNyb3A9Tm9uZSwgZmxpcD1Ob25lKToKICAgIHggPSByYXdb'
    'aWRzXS5mbG9hdCgpLmRpdl8oMjU1KQogICAgaWYgY3JvcCBpcyBub3QgTm9uZToKICAgICAgICB4ID0gRi5wYWQoeCwgKDQsIDQs'
    'IDQsIDQpLCBtb2RlPSdyZWZsZWN0JykKICAgICAgICByb3dzID0gdG9yY2guYXJhbmdlKDMyLCBkZXZpY2U9cmF3LmRldmljZSlb'
    'Tm9uZSwgOl0rY3JvcFs6LCAwLCBOb25lXQogICAgICAgIGNvbHMgPSB0b3JjaC5hcmFuZ2UoMzIsIGRldmljZT1yYXcuZGV2aWNl'
    'KVtOb25lLCA6XStjcm9wWzosIDEsIE5vbmVdCiAgICAgICAgeCA9IHhbdG9yY2guYXJhbmdlKGxlbihpZHMpLCBkZXZpY2U9cmF3'
    'LmRldmljZSlbOiwgTm9uZSwgTm9uZV0sIDosIHJvd3NbOiwgOiwgTm9uZV0sIGNvbHNbOiwgTm9uZSwgOl1dLnBlcm11dGUoMCwg'
    'MywgMSwgMikKICAgICAgICB4ID0gdG9yY2gud2hlcmUoZmxpcFs6LCBOb25lLCBOb25lLCBOb25lXS5ib29sKCksIHguZmxpcCgt'
    'MSksIHgpCiAgICByZXR1cm4gKHgtbWVhbikvc3RkCgoKQHRvcmNoLm5vX2dyYWQoKQpkZWYgZXZhbHVhdGUobW9kZWwsIHJhdywg'
    'bGFiZWxzLCBpZHMsIG1lYW4sIHN0ZCwgc2hpZnRzPUZhbHNlLCBjYXB0dXJlPU5vbmUpOgogICAgbW9kZWwuZXZhbCgpCiAgICB0'
    'b3RhbHMgPSBucC56ZXJvcygoNSBpZiBzaGlmdHMgZWxzZSAxLCAzKSwgZHR5cGU9bnAuZmxvYXQ2NCkKICAgIGRpcmVjdGlvbnMg'
    'PSAoKDAsIDApLCAoLTEsIDApLCAoMSwgMCksICgwLCAtMSksICgwLCAxKSkgaWYgc2hpZnRzIGVsc2UgKCgwLCAwKSwpCiAgICBp'
    'ZiBjYXB0dXJlIGlzIG5vdCBOb25lOgogICAgICAgIGNhcHR1cmUudXBkYXRlKGlkcz1bXSwgbGFiZWxzPVtdLCBsb2dpdHM9W10p'
    'CiAgICBmb3Igc3RhcnQgaW4gcmFuZ2UoMCwgbGVuKGlkcyksIEJBVENIKToKICAgICAgICBpeCA9IGlkc1tzdGFydDpzdGFydCtC'
    'QVRDSF0KICAgICAgICB4ID0gaW5wdXRzKHJhdywgaXgsIG1lYW4sIHN0ZCkKICAgICAgICBiYXRjaF9sb2dpdHMgPSBbXQogICAg'
    'ICAgIGJhc2UgPSBOb25lCiAgICAgICAgZm9yIGksIHNoaWZ0IGluIGVudW1lcmF0ZShkaXJlY3Rpb25zKToKICAgICAgICAgICAg'
    'bG9naXRzID0gbW9kZWwodG9yY2gucm9sbCh4LCBzaGlmdCwgZGltcz0oMiwgMykpIGlmIGkgZWxzZSB4KQogICAgICAgICAgICBw'
    'cmVkID0gbG9naXRzLmFyZ21heCgxKQogICAgICAgICAgICBpZiBjYXB0dXJlIGlzIG5vdCBOb25lOgogICAgICAgICAgICAgICAg'
    'YmF0Y2hfbG9naXRzLmFwcGVuZChsb2dpdHMuY3B1KCkubnVtcHkoKSkKICAgICAgICAgICAgaWYgYmFzZSBpcyBOb25lOgogICAg'
    'ICAgICAgICAgICAgYmFzZSA9IHByZWQKICAgICAgICAgICAgdG90YWxzW2ldICs9IFtGLmNyb3NzX2VudHJvcHkobG9naXRzLCBs'
    'YWJlbHNbaXhdLCByZWR1Y3Rpb249J3N1bScpLml0ZW0oKSwgKHByZWQgPT0gbGFiZWxzW2l4XSkuc3VtKCkuaXRlbSgpLCAocHJl'
    'ZCAhPSBiYXNlKS5zdW0oKS5pdGVtKCldCiAgICAgICAgaWYgY2FwdHVyZSBpcyBub3QgTm9uZToKICAgICAgICAgICAgY2FwdHVy'
    'ZVsnaWRzJ10uYXBwZW5kKGl4LmNwdSgpLm51bXB5KCkpCiAgICAgICAgICAgIGNhcHR1cmVbJ2xhYmVscyddLmFwcGVuZChsYWJl'
    'bHNbaXhdLmNwdSgpLm51bXB5KCkpCiAgICAgICAgICAgIGNhcHR1cmVbJ2xvZ2l0cyddLmFwcGVuZChucC5zdGFjayhiYXRjaF9s'
    'b2dpdHMsIGF4aXM9MSkpCiAgICBpZiBjYXB0dXJlIGlzIG5vdCBOb25lOgogICAgICAgIGNhcHR1cmUudXBkYXRlKHtrOiBucC5j'
    'b25jYXRlbmF0ZSh2KSBmb3IgaywgdiBpbiBjYXB0dXJlLml0ZW1zKCl9KQogICAgaWYgbm90IG5wLmlzZmluaXRlKHRvdGFscyku'
    'YWxsKCk6CiAgICAgICAgcmFpc2UgRmxvYXRpbmdQb2ludEVycm9yKCdub25maW5pdGUgZXZhbHVhdGlvbicpCiAgICBzdGF0cyA9'
    'IFtkaWN0KGNlPWZsb2F0KGEvbGVuKGlkcykpLCBhY2N1cmFjeT1mbG9hdChiL2xlbihpZHMpKSwgZGlzYWdyZWVtZW50PWZsb2F0'
    'KGMvbGVuKGlkcykpKSBmb3IgYSwgYiwgYyBpbiB0b3RhbHNdCiAgICByZXR1cm4gZGljdChjbGVhbj1zdGF0c1swXSwgdHJhbnNs'
    'YXRpb25zPWRpY3QoemlwKCgndXAnLCAnZG93bicsICdsZWZ0JywgJ3JpZ2h0JyksIHN0YXRzWzE6XSkpLCBjb25zaXN0ZW5jeV9l'
    'cnJvcj1mbG9hdCh0b3RhbHNbMTosIDJdLnN1bSgpLyg0KmxlbihpZHMpKSkgaWYgc2hpZnRzIGVsc2UgTm9uZSkKCgpkZWYgc3Vt'
    'bWFyaXplX3BhaXJzKGNvbnNpc3RlbmN5LCBjZSk6CiAgICBpZiBsZW4oY29uc2lzdGVuY3kpICE9IGxlbihTRUVEUykgb3IgbGVu'
    'KGNlKSAhPSBsZW4oU0VFRFMpOgogICAgICAgIHJhaXNlIFZhbHVlRXJyb3IoJ2FsbCBwYWlyZWQgZW5kcG9pbnRzIHJlcXVpcmVk'
    'JykKICAgIGlmIG5vdCBhbGwobWF0aC5pc2Zpbml0ZSh4KSBmb3IgeCBpbiAoKmNvbnNpc3RlbmN5LCAqY2UpKToKICAgICAgICBy'
    'YWlzZSBGbG9hdGluZ1BvaW50RXJyb3IoJ25vbmZpbml0ZSBwYWlyZWQgZW5kcG9pbnQnKQogICAgZGVmIHN1bW1hcnkodmFsdWVz'
    'KToKICAgICAgICBuID0gbGVuKHZhbHVlcykKICAgICAgICBtZWFuID0gc3VtKHZhbHVlcykgLyBuCiAgICAgICAgc2QgPSBtYXRo'
    'LnNxcnQoc3VtKCh4LW1lYW4pKioyIGZvciB4IGluIHZhbHVlcykgLyAobi0xKSkKICAgICAgICBtYXJnaW4gPSAyLjU3MDU4MTgz'
    'NTYzNjMxNCAqIHNkIC8gbWF0aC5zcXJ0KG4pCiAgICAgICAgcmV0dXJuIGRpY3QobWVhbj1tZWFuLCBzYW1wbGVfc2Q9c2QsIGRl'
    'c2NyaXB0aXZlX3Q5NT1bbWVhbi1tYXJnaW4sIG1lYW4rbWFyZ2luXSkKICAgIGNvbnNpc3RlbmN5X3N1bW1hcnksIGNlX3N1bW1h'
    'cnkgPSBzdW1tYXJ5KGNvbnNpc3RlbmN5KSwgc3VtbWFyeShjZSkKICAgIGNsYXVzZXMgPSBkaWN0KG1lYW5fY29uc2lzdGVuY3k9'
    'Y29uc2lzdGVuY3lfc3VtbWFyeVsnbWVhbiddIDw9IC0uMDIsCiAgICAgICAgICAgICAgICAgICBhbGxfY29uc2lzdGVuY3lfbmVn'
    'YXRpdmU9YWxsKHggPCAwIGZvciB4IGluIGNvbnNpc3RlbmN5KSwKICAgICAgICAgICAgICAgICAgIG1lYW5fY2xlYW5fY2U9Y2Vf'
    'c3VtbWFyeVsnbWVhbiddIDw9IC4wMikKICAgIHJldHVybiBjb25zaXN0ZW5jeV9zdW1tYXJ5LCBjZV9zdW1tYXJ5LCBjbGF1c2Vz'
    'CgoKZGVmIG1haW4oKToKICAgIGlmIG5vdCB0b3JjaC5jdWRhLmlzX2F2YWlsYWJsZSgpOgogICAgICAgIHJhaXNlIFJ1bnRpbWVF'
    'cnJvcignQ1VEQSBtYW5kYXRvcnknKQogICAgT1VULm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkKICAgIGlmIChP'
    'VVQvJ2FydGlmYWN0cy5qc29uJykuZXhpc3RzKCkgb3IgKE9VVC8ncHJvZ3Jlc3MuanNvbmwnKS5leGlzdHMoKToKICAgICAgICBy'
    'YWlzZSBSdW50aW1lRXJyb3IoJ2ZyZXNoIG91dHB1dCBkaXJlY3RvcnkgcmVxdWlyZWQnKQogICAgdG9yY2guc2V0X251bV90aHJl'
    'YWRzKDQpCiAgICB0b3JjaC51c2VfZGV0ZXJtaW5pc3RpY19hbGdvcml0aG1zKFRydWUpCiAgICB0b3JjaC5iYWNrZW5kcy5jdWRu'
    'bi5iZW5jaG1hcmsgPSBGYWxzZQogICAgdG9yY2guYmFja2VuZHMuY3VkYS5tYXRtdWwuYWxsb3dfdGYzMiA9IEZhbHNlCiAgICB0'
    'b3JjaC5iYWNrZW5kcy5jdWRubi5hbGxvd190ZjMyID0gRmFsc2UKICAgIGltcG9ydCBkYXRhc2V0cwogICAgaW1wb3J0IGh1Z2dp'
    'bmdmYWNlX2h1YgogICAgZnJvbSBodWdnaW5nZmFjZV9odWIgaW1wb3J0IEhmQXBpLCBoZl9odWJfZG93bmxvYWQKICAgIGltcG9y'
    'dCBuZXVyYWxfbGFiLmFydGlmYWN0cyBhcyBhcnRpZmFjdHMKICAgIHdyaXRlciA9IEFydGlmYWN0V3JpdGVyKE9VVCkKICAgIChP'
    'VVQvJ3Byb3ZlbmFuY2UnKS5ta2RpcigpCiAgICByZXZpc2lvbiA9ICcwYjI3MTQ5ODdmYTQ3ODQ4M2FmOTk2OGRlN2M5MzQ1ODBk'
    'MGJiOWEyJwogICAgbWV0YWRhdGEgPSBIZkFwaSh0b2tlbj1GYWxzZSkuZGF0YXNldF9pbmZvKCd1b2Z0LWNzL2NpZmFyMTAnLCBy'
    'ZXZpc2lvbj1yZXZpc2lvbiwgdGltZW91dD0yMCkKICAgIGlmIG1ldGFkYXRhLnNoYSAhPSByZXZpc2lvbjoKICAgICAgICByYWlz'
    'ZSBWYWx1ZUVycm9yKCdpbW11dGFibGUgSEYgcmV2aXNpb24gbWlzbWF0Y2gnKQogICAgc291cmNlX3BhdGggPSBoZl9odWJfZG93'
    'bmxvYWQocmVwb19pZD0ndW9mdC1jcy9jaWZhcjEwJywgcmVwb190eXBlPSdkYXRhc2V0JywgZmlsZW5hbWU9J3BsYWluX3RleHQv'
    'dHJhaW4tMDAwMDAtb2YtMDAwMDEucGFycXVldCcsIHJldmlzaW9uPXJldmlzaW9uLCB0b2tlbj1GYWxzZSwgbG9jYWxfZGlyPXN0'
    'cihPVVQvJ2hmLXNvdXJjZScpKQogICAgaWYgc2hhKHNvdXJjZV9wYXRoKSAhPSBTT1VSQ0VfU0hBOgogICAgICAgIHJhaXNlIFZh'
    'bHVlRXJyb3IoJ3Bpbm5lZCBvZmZpY2lhbCB0cmFpbmluZyBwYXJxdWV0IGhhc2ggbWlzbWF0Y2gnKQogICAgc2h1dGlsLmNvcHlm'
    'aWxlKHNvdXJjZV9wYXRoLCBPVVQvJ3Byb3ZlbmFuY2UvdHJhaW4ucGFycXVldCcpCiAgICBzaHV0aWwuY29weWZpbGUoX19maWxl'
    'X18sIE9VVC8ncHJvdmVuYW5jZS9leHBlcmltZW50LnB5JykKICAgIHNodXRpbC5jb3B5ZmlsZShhcnRpZmFjdHMuX19maWxlX18s'
    'IE9VVC8ncHJvdmVuYW5jZS9hcnRpZmFjdHMucHknKQogICAgc291cmNlID0gd3JpdGVyLnJlZ2lzdGVyX2ZpbGUoJ2RhdGFzZXRz'
    'JywgT1VULydwcm92ZW5hbmNlL3RyYWluLnBhcnF1ZXQnLCAnb2ZmaWNpYWwtdHJhaW4tcGFycXVldCcsIGZvcm1hdD0ncGFycXVl'
    'dCcsIHNvdXJjZT1kaWN0KHJlcG89J3VvZnQtY3MvY2lmYXIxMCcsIHJldmlzaW9uPScwYjI3MTQ5ODdmYTQ3ODQ4M2FmOTk2OGRl'
    'N2M5MzQ1ODBkMGJiOWEyJywgcGF0aD0ncGxhaW5fdGV4dC90cmFpbi0wMDAwMC1vZi0wMDAwMS5wYXJxdWV0JywgY3JlYXRvcj0n'
    'aHR0cHM6Ly93d3cuY3MudG9yb250by5lZHUvfmtyaXovY2lmYXIuaHRtbCcsIHVybD0naHR0cHM6Ly9odWdnaW5nZmFjZS5jby9k'
    'YXRhc2V0cy91b2Z0LWNzL2NpZmFyMTAvdHJlZS8nK3JldmlzaW9uLCBhdXRoZW50aWNhdGlvbj0ncHVibGljIHRva2VuPUZhbHNl'
    'JywgbGljZW5zZT1tZXRhZGF0YS5jYXJkX2RhdGEuZ2V0KCdsaWNlbnNlJykgaWYgbWV0YWRhdGEuY2FyZF9kYXRhIGVsc2UgTm9u'
    'ZSwgZGF0YXNldHNfdmVyc2lvbj1kYXRhc2V0cy5fX3ZlcnNpb25fXywgaHVnZ2luZ2ZhY2VfaHViX3ZlcnNpb249aHVnZ2luZ2Zh'
    'Y2VfaHViLl9fdmVyc2lvbl9fKSkKICAgIGZvciBmaWxlbmFtZSBpbiAoJ2V4cGVyaW1lbnQucHknLCAnYXJ0aWZhY3RzLnB5Jyk6'
    'CiAgICAgICAgd3JpdGVyLnJlZ2lzdGVyX2ZpbGUoJ2RhdGFzZXRzJywgT1VULydwcm92ZW5hbmNlJy9maWxlbmFtZSwgZmlsZW5h'
    'bWUsIGZvcm1hdD0ncHl0aG9uJywgc291cmNlPWRpY3Qocm9sZT0nZXhlY3V0ZWQgc291cmNlIHNuYXBzaG90JykpCiAgICBkcyA9'
    'IGRhdGFzZXRzLmxvYWRfZGF0YXNldCgncGFycXVldCcsIGRhdGFfZmlsZXM9eyd0cmFpbic6IHN0cihPVVQvJ3Byb3ZlbmFuY2Uv'
    'dHJhaW4ucGFycXVldCcpfSwgc3BsaXQ9J3RyYWluJywgY2FjaGVfZGlyPXN0cihPVVQvJ2NhY2hlJykpCiAgICBpbWFnZXMgPSBu'
    'cC5zdGFjayhbbnAuYXNhcnJheShyb3dbJ2ltZyddLmNvbnZlcnQoJ1JHQicpLCBkdHlwZT1ucC51aW50OCkgZm9yIHJvdyBpbiBk'
    'c10pCiAgICBsYWJlbHMgPSBucC5hc2FycmF5KGRzWydsYWJlbCddLCBkdHlwZT1ucC5pbnQ2NCkKICAgIGlmIGltYWdlcy5zaGFw'
    'ZSAhPSAoNTAwMDAsIDMyLCAzMiwgMykgb3Igbm90IG5wLmFycmF5X2VxdWFsKG5wLmJpbmNvdW50KGxhYmVscyksIG5wLmZ1bGwo'
    'MTAsIDUwMDApKToKICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCd1bmV4cGVjdGVkIENJRkFSIGxheW91dCcpCiAgICBybmcgPSBu'
    'cC5yYW5kb20uZGVmYXVsdF9ybmcoMjAyNjEwMzApCiAgICB0cmFpbiwgZGV2ID0gW10sIFtdCiAgICBmb3IgYyBpbiByYW5nZSgx'
    'MCk6CiAgICAgICAgaW5kaWNlcyA9IHJuZy5wZXJtdXRhdGlvbihucC5mbGF0bm9uemVybyhsYWJlbHMgPT0gYykpCiAgICAgICAg'
    'dHJhaW4uZXh0ZW5kKGluZGljZXNbOjQ1MDBdKTsgZGV2LmV4dGVuZChpbmRpY2VzWzQ1MDA6XSkKICAgIHRyYWluLCBkZXYgPSBu'
    'cC5hc2FycmF5KHNvcnRlZCh0cmFpbikpLCBucC5hc2FycmF5KHNvcnRlZChkZXYpKQogICAgcGl4ZWxzID0gaW1hZ2VzW3RyYWlu'
    'XS5hc3R5cGUobnAuZmxvYXQ2NCkvMjU1CiAgICBtZWFuLCBzdGQgPSBwaXhlbHMubWVhbigoMCwgMSwgMikpLCBwaXhlbHMuc3Rk'
    'KCgwLCAxLCAyKSkKICAgIGRlbCBwaXhlbHMKICAgIGRhdGEgPSB3cml0ZXIuc2F2ZV9kYXRhc2V0KCdzcGxpdCcsIGRpY3QoaW1h'
    'Z2VzPWltYWdlcywgbGFiZWxzPWxhYmVscywgdHJhaW5faWRzPXRyYWluLCBkZXZfaWRzPWRldiwgbWVhbj1tZWFuLCBzdGQ9c3Rk'
    'KSwgcHJvdmVuYW5jZT1kaWN0KHNvdXJjZT1zb3VyY2VbJ3BhdGgnXSwgc3BsaXRfc2VlZD0yMDI2MTAzMCwgbnVtcHk9bnAuX192'
    'ZXJzaW9uX18sIHNwbGl0PSdQQ0c2NCBwZXItY2xhc3MgcGVybXV0YXRpb25zLCBzb3J0ZWQgSURzLCA0NTAwLzUwMCBwZXIgY2xh'
    'c3MnLCBub3JtYWxpemF0aW9uPSd0cmFpbiBSR0IgcG9wdWxhdGlvbiBtb21lbnRzJykpCiAgICByYXcgPSB0b3JjaC5mcm9tX251'
    'bXB5KGltYWdlcy50cmFuc3Bvc2UoMCwgMywgMSwgMikuY29weSgpKS5jdWRhKCkKICAgIHkgPSB0b3JjaC5mcm9tX251bXB5KGxh'
    'YmVscykuY3VkYSgpCiAgICBtdSA9IHRvcmNoLnRlbnNvcihtZWFuLCBkdHlwZT10b3JjaC5mbG9hdDMyLCBkZXZpY2U9J2N1ZGEn'
    'KVtOb25lLCA6LCBOb25lLCBOb25lXQogICAgc2QgPSB0b3JjaC50ZW5zb3Ioc3RkLCBkdHlwZT10b3JjaC5mbG9hdDMyLCBkZXZp'
    'Y2U9J2N1ZGEnKVtOb25lLCA6LCBOb25lLCBOb25lXQogICAgdHJhaW5fZ3B1LCBkZXZfZ3B1ID0gdG9yY2gudGVuc29yKHRyYWlu'
    'LCBkZXZpY2U9J2N1ZGEnKSwgdG9yY2gudGVuc29yKGRldiwgZGV2aWNlPSdjdWRhJykKICAgIGNvbmZpZyA9IGRpY3QoZXBvY2hz'
    'PUVQT0NIUywgc2VlZHM9U0VFRFMsIGFybXM9QVJNUywgc3BsaXRfc2VlZD0yMDI2MTAzMCwgYmF0Y2g9QkFUQ0gsIG9wdGltaXpl'
    'cj0nQWRhbVcgbHI9MC4wMDEgd2VpZ2h0X2RlY2F5PTAuMDEgY29uc3RhbnQnLCBkdHlwZT0nZmxvYXQzMiBkZXRlcm1pbmlzdGlj'
    'LCBURjMyIGRpc2FibGVkJywgaW50ZXJ2ZW50aW9uPSdyZWZsZWN0IHBhZDEgdGhlbiBjaGFubmVsd2lzZSBiaW5vbWlhbCBbMSwy'
    'LDFdIG91dGVyIC8xNiBvbiBub3JtYWxpemVkIGlucHV0IGJlZm9yZSBsZWFybmVkIHN0ZW0nLCB0aHJlc2hvbGQ9J21lYW4gY2Fu'
    'ZGlkYXRlLW1pbnVzLXBsYWluIGNvbnNpc3RlbmN5IDw9IC0wLjAyOyBhbGwgc2l4IHBhaXJlZCBkaWZmZXJlbmNlcyA8MDsgbWVh'
    'biBjbGVhbiBDRSBkaWZmZXJlbmNlIDw9ICswLjAyJywgdHJhbnNsYXRpb249J3RvcmNoLnJvbGwgb25lIHBpeGVsIGVhY2ggY2Fy'
    'ZGluYWwgZGlyZWN0aW9uOyBjaXJjdWxhciBib3VuZGFyeScsIGNoZWNrcG9pbnRfc2VsZWN0aW9uPSdmaW5hbCBlbmRwb2ludDsg'
    'QmVzdENoZWNrcG9pbnQgZWFybGllc3QgbWluaW11bSBvZiBwb3N0LWVwb2NoIGNhbm9uaWNhbCB0cmFpbiBpbmZlcmVuY2UgQ0Un'
    'KQogICAgcmVjb3JkcyA9IFtdCiAgICBmb3Igc2VlZCBpbiBTRUVEUzoKICAgICAgICBvcmRlciwgY3JvcCwgZmxpcCA9IHNjaGVk'
    'dWxlKHNlZWQsIGxlbih0cmFpbikpCiAgICAgICAgc2NoZWQgPSB3cml0ZXIuc2F2ZV9kYXRhc2V0KGYnc2NoZWR1bGUte3NlZWR9'
    'JywgZGljdChvcmRlcj1vcmRlciwgY3JvcD1jcm9wLCBmbGlwPWZsaXApLCBwcm92ZW5hbmNlPWRpY3Qoc2VlZD1zZWVkLCBudW1w'
    'eT1ucC5fX3ZlcnNpb25fXywgZ2VuZXJhdG9yPSdQQ0c2NCBwZXIgZXBvY2ggc2VlZCsxMDAwMDArZXBvY2gqMTAwMDAwMywgb3Jk'
    'ZXIgdGhlbiBjcm9wIHRoZW4gZmxpcCcsIGluZGV4aW5nPSdlcG9jaCxwb3NpdGlvbjsgb3JkZXIgaW5kZXhlcyBzb3J0ZWQgdHJh'
    'aW5faWRzJykpCiAgICAgICAgdG9yY2gubWFudWFsX3NlZWQoc2VlZCk7IHRvcmNoLmN1ZGEubWFudWFsX3NlZWRfYWxsKHNlZWQp'
    'CiAgICAgICAgaW5pdGlhbCA9IGNvcHkuZGVlcGNvcHkoTW9kZWwoJ3BsYWluJykuc3RhdGVfZGljdCgpKQogICAgICAgIGFybV9v'
    'cmRlciA9IEFSTVMgaWYgc2VlZCAlIDIgZWxzZSBBUk1TWzo6LTFdCiAgICAgICAgZm9yIGFybSBpbiBhcm1fb3JkZXI6CiAgICAg'
    'ICAgICAgIG1vZGVsID0gTW9kZWwoYXJtKQogICAgICAgICAgICBzdGF0ZSA9IG1vZGVsLnN0YXRlX2RpY3QoKQogICAgICAgICAg'
    'ICBzdGF0ZS51cGRhdGUoe2s6IHYuY2xvbmUoKSBmb3IgaywgdiBpbiBpbml0aWFsLml0ZW1zKCl9KQogICAgICAgICAgICBtb2Rl'
    'bC5sb2FkX3N0YXRlX2RpY3Qoc3RhdGUsIHN0cmljdD1UcnVlKQogICAgICAgICAgICBhc3NlcnQgYWxsKHRvcmNoLmVxdWFsKG1v'
    'ZGVsLnN0YXRlX2RpY3QoKVtrXSwgdikgZm9yIGssIHYgaW4gaW5pdGlhbC5pdGVtcygpKQogICAgICAgICAgICByZWZzID0gW3Nv'
    'dXJjZVsncGF0aCddLCBkYXRhWydwYXRoJ10sIHNjaGVkWydwYXRoJ11dCiAgICAgICAgICAgIGluaXRpYWxfcmVmID0gd3JpdGVy'
    'LnNhdmVfY2hlY2twb2ludChmJ3tzZWVkfS17YXJtfS1pbml0aWFsJywgbW9kZWwsIGFybT1hcm0sIHNlZWQ9c2VlZCwgc2VsZWN0'
    'aW9uPWRpY3QobWV0cmljPSdwYWlyZWQgaW5pdGlhbGl6YXRpb24gYmVmb3JlIHRyYWluaW5nJywgc3RlcD0wKSwgY29uZmlnPWNv'
    'bmZpZywgZGF0YXNldHM9cmVmcykKICAgICAgICAgICAgbW9kZWwuY3VkYSgpCiAgICAgICAgICAgIG9wdGltaXplciA9IHRvcmNo'
    'Lm9wdGltLkFkYW1XKG1vZGVsLnBhcmFtZXRlcnMoKSwgbHI9LjAwMSwgd2VpZ2h0X2RlY2F5PS4wMSkKICAgICAgICAgICAgYmVz'
    'dCA9IEJlc3RDaGVja3BvaW50KHNjb3BlPSdlYXJsaWVzdCBtaW5pbXVtIHBvc3QtZXBvY2ggY2Fub25pY2FsIHRyYWluaW5nIGlu'
    'ZmVyZW5jZSBDRTsgbm8gZGV2IHNlbGVjdGlvbicpCiAgICAgICAgICAgIHRyYWNlcywgZHVyYXRpb25zID0gW10sIFtdCiAgICAg'
    'ICAgICAgIHRvcmNoLmN1ZGEucmVzZXRfcGVha19tZW1vcnlfc3RhdHMoKQogICAgICAgICAgICB0b3JjaC5jdWRhLnN5bmNocm9u'
    'aXplKCk7IHN0YXJ0ZWQgPSB0aW1lLm1vbm90b25pYygpCiAgICAgICAgICAgIHN0ZXAgPSAwCiAgICAgICAgICAgIGZvciBlcG9j'
    'aCBpbiByYW5nZShFUE9DSFMpOgogICAgICAgICAgICAgICAgbW9kZWwudHJhaW4oKTsgdGljayA9IHRpbWUubW9ub3RvbmljKCkK'
    'ICAgICAgICAgICAgICAgIGZvciBzdGFydCBpbiByYW5nZSgwLCBsZW4odHJhaW4pLCBCQVRDSCk6CiAgICAgICAgICAgICAgICAg'
    'ICAgc3RvcCA9IG1pbihzdGFydCtCQVRDSCwgbGVuKHRyYWluKSkKICAgICAgICAgICAgICAgICAgICBpeCA9IHRyYWluX2dwdVt0'
    'b3JjaC5hc190ZW5zb3Iob3JkZXJbZXBvY2gsIHN0YXJ0OnN0b3BdLmFzdHlwZShucC5pbnQ2NCksIGRldmljZT0nY3VkYScpXQog'
    'ICAgICAgICAgICAgICAgICAgIGMgPSB0b3JjaC5hc190ZW5zb3IoY3JvcFtlcG9jaCwgc3RhcnQ6c3RvcF0uYXN0eXBlKG5wLmlu'
    'dDY0KSwgZGV2aWNlPSdjdWRhJykKICAgICAgICAgICAgICAgICAgICBmID0gdG9yY2guYXNfdGVuc29yKGZsaXBbZXBvY2gsIHN0'
    'YXJ0OnN0b3BdLCBkZXZpY2U9J2N1ZGEnKQogICAgICAgICAgICAgICAgICAgIG9wdGltaXplci56ZXJvX2dyYWQoc2V0X3RvX25v'
    'bmU9VHJ1ZSkKICAgICAgICAgICAgICAgICAgICBsb3NzID0gRi5jcm9zc19lbnRyb3B5KG1vZGVsKGlucHV0cyhyYXcsIGl4LCBt'
    'dSwgc2QsIGMsIGYpKSwgeVtpeF0pCiAgICAgICAgICAgICAgICAgICAgaWYgbm90IG1hdGguaXNmaW5pdGUoZmxvYXQobG9zcy5k'
    'ZXRhY2goKSkpOgogICAgICAgICAgICAgICAgICAgICAgICByYWlzZSBGbG9hdGluZ1BvaW50RXJyb3IoJ25vbmZpbml0ZSBsb3Nz'
    'JykKICAgICAgICAgICAgICAgICAgICBsb3NzLmJhY2t3YXJkKCkKICAgICAgICAgICAgICAgICAgICBmb3IgcCBpbiBtb2RlbC5w'
    'YXJhbWV0ZXJzKCk6CiAgICAgICAgICAgICAgICAgICAgICAgIGlmIHAuZ3JhZCBpcyBub3QgTm9uZToKICAgICAgICAgICAgICAg'
    'ICAgICAgICAgICAgIHRvcmNoLl9hc3NlcnRfYXN5bmModG9yY2guaXNmaW5pdGUocC5ncmFkKS5hbGwoKSwgJ25vbmZpbml0ZSBn'
    'cmFkaWVudCcpCiAgICAgICAgICAgICAgICAgICAgb3B0aW1pemVyLnN0ZXAoKTsgc3RlcCArPSAxCiAgICAgICAgICAgICAgICB0'
    'b3JjaC5jdWRhLnN5bmNocm9uaXplKCk7IGR1cmF0aW9ucy5hcHBlbmQodGltZS5tb25vdG9uaWMoKS10aWNrKQogICAgICAgICAg'
    'ICAgICAgdHJhaW5fZXZhbCA9IGV2YWx1YXRlKG1vZGVsLCByYXcsIHksIHRyYWluX2dwdSwgbXUsIHNkKVsnY2xlYW4nXQogICAg'
    'ICAgICAgICAgICAgYmVzdC5vYnNlcnZlKG1vZGVsLCB0cmFpbl9ldmFsWydjZSddLCBzdGVwKQogICAgICAgICAgICAgICAgdHJh'
    'Y2VzLmFwcGVuZChbZXBvY2grMSwgc3RlcCwgdHJhaW5fZXZhbFsnY2UnXSwgdHJhaW5fZXZhbFsnYWNjdXJhY3knXV0pCiAgICAg'
    'ICAgICAgICAgICBldmVudCgnZXBvY2gnLCBzZWVkPXNlZWQsIGFybT1hcm0sIGVwb2NoPWVwb2NoKzEsIHN0ZXA9c3RlcCwgdHJh'
    'aW5fY2U9dHJhaW5fZXZhbFsnY2UnXSkKICAgICAgICAgICAgdG9yY2guY3VkYS5zeW5jaHJvbml6ZSgpOyB0cmFpbmluZ19zZWNv'
    'bmRzID0gdGltZS5tb25vdG9uaWMoKS1zdGFydGVkCiAgICAgICAgICAgIGlmIG5vdCBhbGwodG9yY2guaXNmaW5pdGUodCkuYWxs'
    'KCkuaXRlbSgpIGZvciB0IGluIG1vZGVsLnN0YXRlX2RpY3QoKS52YWx1ZXMoKSk6CiAgICAgICAgICAgICAgICByYWlzZSBGbG9h'
    'dGluZ1BvaW50RXJyb3IoJ25vbmZpbml0ZSBzdGF0ZScpCiAgICAgICAgICAgIGNhcHR1cmVkID0ge30KICAgICAgICAgICAgZGV2'
    'X2V2YWwgPSBldmFsdWF0ZShtb2RlbCwgcmF3LCB5LCBkZXZfZ3B1LCBtdSwgc2QsIHNoaWZ0cz1UcnVlLCBjYXB0dXJlPWNhcHR1'
    'cmVkKQogICAgICAgICAgICB3cml0ZXIuc2F2ZV9kYXRhc2V0KGYne3NlZWR9LXthcm19LWRldi1sb2dpdHMnLCBjYXB0dXJlZCwg'
    'cHJvdmVuYW5jZT1kaWN0KG9yZGVyPVsnY2xlYW4nLCAndXAnLCAnZG93bicsICdsZWZ0JywgJ3JpZ2h0J10sIGNoZWNrcG9pbnQ9'
    'J2ZpeGVkIGZpbmFsIHVwZGF0ZScsIHNlbWFudGljcz0nbG9naXRzIHNoYXBlIE4sNSwxMDsgaWRzIHNvcnRlZCcpKQogICAgICAg'
    'ICAgICBmaW5hbCA9IHdyaXRlci5zYXZlX2NoZWNrcG9pbnQoZid7c2VlZH0te2FybX0tZmluYWwnLCBtb2RlbCwgYXJtPWFybSwg'
    'c2VlZD1zZWVkLCBzZWxlY3Rpb249ZGljdChtZXRyaWM9J2ZpeGVkIGZpbmFsIHVwZGF0ZScsIHN0ZXA9c3RlcCksIGNvbmZpZz1j'
    'b25maWcsIGRhdGFzZXRzPXJlZnMpCiAgICAgICAgICAgIHNlbGVjdGVkID0gYmVzdC5zYXZlKHdyaXRlciwgZid7c2VlZH0te2Fy'
    'bX0tdHJhaW5pbmctc2VsZWN0ZWQnLCBhcm09YXJtLCBzZWVkPXNlZWQsIGNvbmZpZz1jb25maWcsIGRhdGFzZXRzPXJlZnMpCiAg'
    'ICAgICAgICAgIHdyaXRlci5zYXZlX2RhdGFzZXQoZid7c2VlZH0te2FybX0tdHJhaW4tdHJhY2UnLCBkaWN0KG9ic2VydmF0aW9u'
    'cz1ucC5hc2FycmF5KHRyYWNlcykpLCBwcm92ZW5hbmNlPWRpY3QoY29sdW1ucz1bJ2Vwb2NoJywgJ3N0ZXAnLCAnY2Fub25pY2Fs'
    'X3RyYWluX2NlJywgJ2Nhbm9uaWNhbF90cmFpbl9hY2N1cmFjeSddLCBzZWxlY3RlZD1zZWxlY3RlZFsncGF0aCddKSkKICAgICAg'
    'ICAgICAgcmVjb3JkID0gZGljdChzZWVkPXNlZWQsIGFybT1hcm0sIGFybV9vcmRlcj1hcm1fb3JkZXIsIHVwZGF0ZXM9c3RlcCwg'
    'ZXBvY2hzPUVQT0NIUywgcGFyYW1ldGVycz1zdW0ocC5udW1lbCgpIGZvciBwIGluIG1vZGVsLnBhcmFtZXRlcnMoKSksIHRyYWlu'
    'YWJsZV9wYXJhbWV0ZXJzPXN1bShwLm51bWVsKCkgZm9yIHAgaW4gbW9kZWwucGFyYW1ldGVycygpIGlmIHAucmVxdWlyZXNfZ3Jh'
    'ZCksIHRyYWluaW5nX3NlY29uZHNfaW5jbHVkaW5nX3RyYWluX2V2YWw9dHJhaW5pbmdfc2Vjb25kcywgZXBvY2hfdXBkYXRlX3Nl'
    'Y29uZHM9ZHVyYXRpb25zLCBwZWFrX2FsbG9jYXRlZF9ieXRlcz10b3JjaC5jdWRhLm1heF9tZW1vcnlfYWxsb2NhdGVkKCksIHBl'
    'YWtfcmVzZXJ2ZWRfYnl0ZXM9dG9yY2guY3VkYS5tYXhfbWVtb3J5X3Jlc2VydmVkKCksIGZpbmFsX3RyYWluPXRyYWNlc1stMV1b'
    'MjpdLCBmaW5hbF9kZXY9ZGV2X2V2YWwsIGluaXRpYWw9aW5pdGlhbF9yZWZbJ3BhdGgnXSwgZmluYWw9ZmluYWxbJ3BhdGgnXSwg'
    'c2VsZWN0ZWQ9c2VsZWN0ZWRbJ3BhdGgnXSkKICAgICAgICAgICAgcmVjb3Jkcy5hcHBlbmQocmVjb3JkKTsgZXZlbnQoJ2ZpdF9j'
    'b21wbGV0ZScsICoqcmVjb3JkKQogICAgICAgICAgICBkZWwgbW9kZWwsIG9wdGltaXplciwgYmVzdAogICAgZGVmIGxvb2t1cChz'
    'ZWVkLCBhcm0pOgogICAgICAgIHJldHVybiBuZXh0KHJbJ2ZpbmFsX2RldiddIGZvciByIGluIHJlY29yZHMgaWYgclsnc2VlZCdd'
    'ID09IHNlZWQgYW5kIHJbJ2FybSddID09IGFybSkKICAgIGNvbnNpc3RlbmN5ID0gW2xvb2t1cChzLCAnYmlub21pYWwnKVsnY29u'
    'c2lzdGVuY3lfZXJyb3InXS1sb29rdXAocywgJ3BsYWluJylbJ2NvbnNpc3RlbmN5X2Vycm9yJ10gZm9yIHMgaW4gU0VFRFNdCiAg'
    'ICBjZSA9IFtsb29rdXAocywgJ2Jpbm9taWFsJylbJ2NsZWFuJ11bJ2NlJ10tbG9va3VwKHMsICdwbGFpbicpWydjbGVhbiddWydj'
    'ZSddIGZvciBzIGluIFNFRURTXQogICAgY29uc2lzdGVuY3lfc3VtbWFyeSwgY2Vfc3VtbWFyeSwgY2xhdXNlcyA9IHN1bW1hcml6'
    'ZV9wYWlycyhjb25zaXN0ZW5jeSwgY2UpCiAgICBkZXRhaWxzID0gZGljdChjb25maWc9Y29uZmlnLCByZWNvcmRzPXJlY29yZHMs'
    'IHBhaXJlZF9jb25zaXN0ZW5jeT1jb25zaXN0ZW5jeSwgcGFpcmVkX2NsZWFuX2NlPWNlLCBjb25zaXN0ZW5jeV9zdW1tYXJ5PWNv'
    'bnNpc3RlbmN5X3N1bW1hcnksIGNsZWFuX2NlX3N1bW1hcnk9Y2Vfc3VtbWFyeSwgY2xhdXNlcz1jbGF1c2VzLCBzY3JlZW5fcGFz'
    'cz1hbGwoY2xhdXNlcy52YWx1ZXMoKSksIHNvdXJjZV9zaGEyNTY9c2hhKF9fZmlsZV9fKSwgYXJ0aWZhY3RfaGVscGVyX3NoYTI1'
    'Nj1zaGEoYXJ0aWZhY3RzLl9fZmlsZV9fKSwgdG9yY2hfdmVyc2lvbj10b3JjaC5fX3ZlcnNpb25fXywgY3VkYV92ZXJzaW9uPXRv'
    'cmNoLnZlcnNpb24uY3VkYSwgY3Vkbm5fdmVyc2lvbj10b3JjaC5iYWNrZW5kcy5jdWRubi52ZXJzaW9uKCksIG51bXB5X3ZlcnNp'
    'b249bnAuX192ZXJzaW9uX18sIGRhdGFzZXRzX3ZlcnNpb249ZGF0YXNldHMuX192ZXJzaW9uX18sIGh1Z2dpbmdmYWNlX2h1Yl92'
    'ZXJzaW9uPWh1Z2dpbmdmYWNlX2h1Yi5fX3ZlcnNpb25fXywgZGV2aWNlPXRvcmNoLmN1ZGEuZ2V0X2RldmljZV9uYW1lKCksIGVs'
    'YXBzZWRfc2Vjb25kcz10aW1lLm1vbm90b25pYygpLVNUQVJUKQogICAgKE9VVC8nZGV0YWlscy5qc29uJykud3JpdGVfdGV4dChq'
    'c29uLmR1bXBzKGRldGFpbHMsIGluZGVudD0yLCBhbGxvd19uYW49RmFsc2UpKQogICAgKE9VVC8ncmVzdWx0Lmpzb24nKS53cml0'
    'ZV90ZXh0KGpzb24uZHVtcHMoZGljdChtZXRyaWNzPWRpY3Qoc2NyZWVuX3Bhc3M9aW50KGFsbChjbGF1c2VzLnZhbHVlcygpKSks'
    'IG1lYW5fY29uc2lzdGVuY3lfZGlmZmVyZW5jZT1jb25zaXN0ZW5jeV9zdW1tYXJ5WydtZWFuJ10sIG1lYW5fY2xlYW5fY2VfZGlm'
    'ZmVyZW5jZT1jZV9zdW1tYXJ5WydtZWFuJ10sIGZpdHM9bGVuKHJlY29yZHMpKSwgc3VtbWFyeT0nRnJvemVuIHBhaXJlZCBhbnRp'
    'LWFsaWFzIHNjcmVlbjsgaW5kZXBlbmRlbnRseSBhdWRpdCBiZWZvcmUgaW50ZXJwcmV0YXRpb24uJyksIGluZGVudD0yLCBhbGxv'
    'd19uYW49RmFsc2UpKQoKCmlmIF9fbmFtZV9fID09ICdfX21haW5fXyc6CiAgICB0cnk6CiAgICAgICAgbWFpbigpCiAgICBleGNl'
    'cHQgRXhjZXB0aW9uOgogICAgICAgIE9VVC5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29rPVRydWUpCiAgICAgICAgKE9VVC8n'
    'ZmFpbHVyZS10cmFjZWJhY2sudHh0Jykud3JpdGVfdGV4dCh0cmFjZWJhY2suZm9ybWF0X2V4YygpKQogICAgICAgIHJhaXNlCg=='
)
source_bytes = base64.b64decode(EMBEDDED_SOURCE, validate=True)
assert hashlib.sha256(source_bytes).hexdigest() == EXPECTED
sci = types.ModuleType('six_seed_antialias')
sci.__file__ = 'embedded-cifar-antialias-six-seed.py'
exec(compile(source_bytes, sci.__file__, 'exec'), sci.__dict__)
checks = []

def passed(name):
    checks.append(name)
    print(name, flush=True)

# Actual prefilter against a scalar reference, including nonconstant reflected boundaries.
m = sci.Model('binomial').eval()
x = torch.arange(3 * 5 * 6, dtype=torch.float32).reshape(1, 3, 5, 6) / 7
actual = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='reflect'), m.kernel, groups=3)
w = (1, 2, 1)
ref = torch.zeros_like(x)
def reflect(i, size):
    return -i if i < 0 else 2 * size - 2 - i if i >= size else i
for c in range(3):
    for y in range(5):
        for z in range(6):
            ref[0, c, y, z] = sum(w[a]*w[b]*x[0, c, reflect(y+a-1, 5), reflect(z+b-1, 6)].item() for a in range(3) for b in range(3))/16
assert torch.allclose(actual, ref, atol=2e-6, rtol=1e-6)
# Capture the actual model forward input to its learned stem.
captured = []
hook = m.stem[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
with torch.no_grad():
    m(x)
hook.remove()
assert len(captured) == 1 and torch.equal(captured[0], actual)
passed('nonconstant reflect convolution reference and model placement')

# Test actual crop, flip, normalization against independent explicit reflect index arithmetic.
raw = (torch.arange(3*3*32*32, dtype=torch.int64).reshape(3, 3, 32, 32) % 256).to(torch.uint8)
ids = torch.tensor([2, 0, 1]); crops = torch.tensor([[0, 8], [4, 4], [8, 0]])
flips = torch.tensor([0, 1, 1]); mu = torch.tensor([.1, .2, .3]).view(1, 3, 1, 1); sd = torch.tensor([.5, .4, .3]).view(1, 3, 1, 1)
out = sci.inputs(raw, ids, mu, sd, crops, flips)
expected = torch.empty_like(out)
for n in range(3):
    for y in range(32):
        for z in range(32):
            source_z = 31-z if flips[n] else z
            yy = reflect(y + int(crops[n, 0]) - 4, 32)
            zz = reflect(source_z + int(crops[n, 1]) - 4, 32)
            expected[n, :, y, z] = (raw[ids[n], :, yy, zz].float()/255-mu[0,:,0,0])/sd[0,:,0,0]
assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)
passed('crop flip normalization reference')

# Small n tests all epoch schedules and permutation/position alignment, reproducibility and seed variation.
a = sci.schedule(3801, 17); b = sci.schedule(3801, 17); c = sci.schedule(3802, 17)
assert all(np.array_equal(p, q) for p, q in zip(a, b))
assert all(np.array_equal(np.sort(row), np.arange(17)) for row in a[0])
assert a[1].shape == (sci.EPOCHS, 17, 2) and a[2].shape == (sci.EPOCHS, 17)
assert np.all((a[1] >= 0) & (a[1] <= 8)) and np.all(a[2] <= 1)
assert not np.array_equal(a[0], c[0])
passed('paired schedule reproducibility and bounds')

# Actual evaluation, five directions, ties, weighted CE and a final partial batch.
# Synthetic model exposes specific corner pixels to make wrapping and directions observable.
class Probe(nn.Module):
    def forward(self, x):
        return torch.stack((x[:, 0, 0, 0], x[:, 0, 0, 1], torch.zeros_like(x[:, 0, 0, 0])), 1)
N = sci.BATCH + 3
raw = torch.zeros((N, 3, 32, 32), dtype=torch.uint8)
for i in range(N):
    raw[i, 0, 0, 0] = (i % 3) * 30
    raw[i, 0, 0, 1] = (i % 4) * 20
    raw[i, 0, 1, 0] = (i % 5) * 15
    raw[i, 0, 31, 0] = (i % 6) * 10
    raw[i, 0, 0, 31] = (i % 7) * 8
labels = torch.arange(N) % 3
ids = torch.arange(N); capture = {}
result = sci.evaluate(Probe(), raw, labels, ids, torch.zeros(1,3,1,1), torch.ones(1,3,1,1), shifts=True, capture=capture)
assert np.array_equal(capture['ids'], np.arange(N)) and np.array_equal(capture['labels'], labels.numpy())
assert capture['logits'].shape == (N, 5, 3)
dirs = [(0,0),(-1,0),(1,0),(0,-1),(0,1)]
logits = []
for dy, dx in dirs:
    # Independently index the source location for destination (0,0) and (0,1).
    v0 = raw[:, 0, (-dy)%32, (-dx)%32].float()/255
    v1 = raw[:, 0, (-dy)%32, (1-dx)%32].float()/255
    logits.append(torch.stack((v0, v1, torch.zeros_like(v0)), 1))
logits = torch.stack(logits, 1)
assert torch.equal(torch.from_numpy(capture['logits']), logits)
preds = logits.argmax(2); assert preds[0,0].item() == 0  # all-zero tie
stats = [result['clean']] + list(result['translations'].values())
for j, stat in enumerate(stats):
    assert abs(stat['ce'] - F.cross_entropy(logits[:,j], labels, reduction='sum').item()/N) < 1e-6
    assert abs(stat['accuracy'] - (preds[:,j] == labels).sum().item()/N) < 1e-12
    assert abs(stat['disagreement'] - (preds[:,j] != preds[:,0]).sum().item()/N) < 1e-12
assert abs(result['consistency_error'] - (preds[:,1:] != preds[:,0,None]).sum().item()/(4*N)) < 1e-12
passed('endpoint directions ties counts CE and partial batch')

# Descriptive selected checkpoint: earliest equal minimum, full tensor round-trip and outputs.
with tempfile.TemporaryDirectory() as d:
    writer = ArtifactWriter(d)
    for arm in sci.ARMS:
        model = sci.Model(arm).eval()
        with torch.no_grad():
            model.stem[2].running_mean.fill_(.25)
            model.stem[2].num_batches_tracked.fill_(7)
        best = BestCheckpoint()
        best.observe(model, 2., 1); best.observe(model, 1., 2); best.observe(model, 1., 3)
        assert best.step == 2
        entry = best.save(writer, arm+'-selected', arm=arm, seed=3801)
        final_entry = writer.save_checkpoint(arm+'-final', model, arm=arm, seed=3801, selection=dict(metric='fixed final update', step=3))
        restored = sci.Model(arm).eval()
        restored.load_state_dict(load_json_state(Path(d)/entry['path']), strict=True)
        assert all(torch.equal(v, restored.state_dict()[k]) for k, v in best.state.items())
        final_restored = sci.Model(arm).eval()
        final_restored.load_state_dict(load_json_state(Path(d)/final_entry['path']), strict=True)
        assert all(torch.equal(v, final_restored.state_dict()[k]) for k, v in model.state_dict().items())
        z = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            assert torch.equal(model(z), restored(z))
            assert torch.equal(model(z), final_restored(z))
passed('selected and final checkpoint full state roundtrip')

# Six-pair decision boundaries and invalid input handling.
cs, ce, clauses = sci.summarize_pairs([-.02]*6, [.02]*6)
assert all(clauses.values())
assert not sci.summarize_pairs([-.03]*5+[0.], [0.]*6)[2]['all_consistency_negative']
assert not sci.summarize_pairs([-.01]*6, [0.]*6)[2]['mean_consistency']
assert not sci.summarize_pairs([-.03]*6, [.03]*6)[2]['mean_clean_ce']
for a, b in (([-.02]*5, [0.]*6), ([-.02]*5+[float('nan')], [0.]*6)):
    try:
        sci.summarize_pairs(a, b)
        raise AssertionError('invalid endpoints accepted')
    except (ValueError, FloatingPointError):
        pass
passed('six-pair decision thresholds and invalid inputs')

outdir = Path('/output'); outdir.mkdir(parents=True, exist_ok=True)
(outdir/'diagnostic-result.json').write_text(json.dumps(dict(science_sha256=EXPECTED, checks=checks, device='cpu', no_training=True, no_dataset=True), indent=2))

(outdir/'result.json').write_text(json.dumps(dict(metrics=dict(checks_passed=len(checks)), summary='Synthetic CPU diagnostic completed; no training or development data used.'), indent=2))
