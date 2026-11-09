import unittest
from collections import Counter
from research.focus_recovery import focus_cases,score,RETENTION
from research.mastery_core import c
class FocusContracts(unittest.TestCase):
    def test_focus_and_retention_distribution(self):
        cases=focus_cases();self.assertEqual(len(cases),1024)
        self.assertEqual(Counter(s.width for s in cases),{440:204,520:307,640:513})
        for width in (440,520,640):
            group=[s for s in cases if s.width==width]
            long=sum(s.scene.pattern=='long' for s in group)
            self.assertAlmostEqual(long/len(group),.5,delta=.005)
            self.assertTrue(all((s.scene.chasers,s.scene.shooters) not in c.HOLDOUT for s in group))
            self.assertTrue(all(s.scene.pattern!='unequal' for s in group))
            self.assertGreaterEqual(sum(s.scene in RETENTION for s in group)/len(group),.3)
if __name__=='__main__':unittest.main()
