import unittest
from unittest.mock import patch
from collect_source_observations import drive_inventory
class SourceObservationTests(unittest.TestCase):
 def test_drive_inventory_is_bounded_and_explicitly_partial(self):
  counter=[]
  def request(*a):counter.append(1);return {'files':[{'id':'x'}],'nextPageToken':str(len(counter))}
  with patch.dict('os.environ',{'DRIVE_TOKEN':'secret'}):r=drive_inventory({'folder_id':'folder','token_env':'DRIVE_TOKEN'},request)
  self.assertEqual(r['count'],10);self.assertFalse(r['complete']);self.assertEqual(r['reason'],'page_limit')
