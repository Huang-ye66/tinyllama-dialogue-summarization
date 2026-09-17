import unittest
from scripts.long_dialogue.hierarchical import chunk_dialogue, source_facts, collapse_repeated_dialogue, repair_attribution_conflicts
from scripts.core.reliability import assess_summary

class WordTokenizer:
    def encode(self, text):
        return text.split()
    def decode(self, ids):
        return " ".join(ids)

class LongDialogueTests(unittest.TestCase):
    def setUp(self):
        self.tok=WordTokenizer()

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            chunk_dialogue(self.tok,"")

    def test_complete_turn_coverage(self):
        dialogue="\n".join(f"P{i}: "+("word "*35) for i in range(9))
        chunks=chunk_dialogue(self.tok,dialogue,budget=100)
        covered={i for chunk in chunks for i in chunk.turn_indices}
        self.assertEqual(covered,set(range(9)))
        self.assertTrue(all(chunk.token_count<=100 for chunk in chunks))

    def test_single_long_turn_uses_windows(self):
        dialogue="Alice: "+("word "*260)
        chunks=chunk_dialogue(self.tok,dialogue,budget=100)
        self.assertGreater(len(chunks),1)
        self.assertTrue(all(chunk.turn_indices==[0] for chunk in chunks))
        self.assertTrue(all(chunk.token_count<=100 for chunk in chunks))

    def test_sentence_first_split(self):
        dialogue="Alice: "+("one "*60)+". "+("two "*60)+"."
        chunks=chunk_dialogue(self.tok,dialogue,budget=80)
        self.assertEqual(len(chunks),2)
        self.assertTrue(all(chunk.turn_indices==[0] for chunk in chunks))

    def test_unicode_and_facts(self):
        dialogue="Alice: Meet Bob at 3:00 PM.\nJosé: Fine, the budget is €10."
        facts=source_facts(dialogue)
        self.assertIn("Alice",facts["speakers"])
        self.assertIn("Bob",facts["mentioned_people"])
        self.assertTrue(facts["numbers"])

    def test_repeated_block_collapsed_with_provenance(self):
        block=["Alice: Plan launch.","Bob: Call clients.","Carol: Fix site.","David: Test site."]
        cleaned,provenance=collapse_repeated_dialogue("\n".join(block*3))
        self.assertEqual(cleaned,"\n".join(block))
        self.assertEqual(provenance[0],[0,4,8])

    def test_single_repeated_turn_is_not_removed(self):
        dialogue="Alice: Okay.\nAlice: Okay.\nBob: Continue."
        cleaned,_=collapse_repeated_dialogue(dialogue)
        self.assertEqual(cleaned,dialogue)

    def test_high_confidence_attribution_repair(self):
        dialogue="Alice: Bob must finish the client list.\nBob: I will finish the client list.\nDavid: I will test the payment page with Carol and record failed cases.\nCarol: I will fix the payment page."
        summary="Bob will test the payment page with Carol and record failed cases. David will test the payment page with Carol and record failed cases."
        repaired,changes=repair_attribution_conflicts(dialogue,summary)
        self.assertNotIn("Bob will test",repaired)
        self.assertIn("David will test",repaired)
        self.assertTrue(any(x.get("reason")=="unsupported_explicit_attribution" for x in changes))
        self.assertTrue(any(x.get("reason")=="near_duplicate_event" for x in changes))

    def test_single_unsupported_explicit_actions_are_corrected(self):
        dialogue="Carol: The payment page needs a fix.\nDavid: I will test the payment page with Carol.\nDavid: I can grant Carol access to the logs.\nBob: I will contact clients."
        summary="Carol will test the payment page. Bob will grant Carol access to the logs."
        repaired,changes=repair_attribution_conflicts(dialogue,summary)
        self.assertIn("David will test",repaired)
        self.assertIn("David will grant",repaired)
        self.assertEqual(sum(x.get("reason")=="unsupported_explicit_attribution" for x in changes),2)

    def test_reliability_detects_incomplete_and_attribution_conflict(self):
        dialogue="Alice: I will prepare the list.\nBob: I will test the page.\nDavid: I will grant access."
        summary="Bob will test the page and record failures. David will test the page and record failures. Alice needs"
        result=assess_summary(dialogue,summary)
        self.assertTrue(result["flags"]["possible_attribution_conflict"])
        self.assertTrue(result["flags"]["incomplete_sentence"])
        self.assertEqual(result["risk"],"high")

if __name__=="__main__":
    unittest.main()
