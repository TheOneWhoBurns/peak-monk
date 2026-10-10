import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('prompt_exclusion',Path(__file__).parents[1]/'scripts/audit_prompt_exclusion.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)


def test_known_caller_collision_and_assistant_noncollision(tmp_path):
    suite=tmp_path/'suite.json';corpus=tmp_path/'data.jsonl'
    suite.write_text(json.dumps([{'id':'hidden','turns':['Where is the blue box?']}]))
    corpus.write_text(json.dumps({'messages':[{'role':'user','content':'WHERE is the blue box!'}, {'role':'assistant','content':'Here.'}]}))
    report=checker.audit([suite],[corpus])
    assert report['status']=='fail' and len(report['collisions'])==1
    assert report['unique_evaluation_callers']==1
    corpus.write_text(json.dumps({'messages':[{'role':'user','content':'A different question.'},{'role':'assistant','content':'Where is the blue box?'}]}))
    assert checker.audit([suite],[corpus])['status']=='pass'


def test_reject_empty_and_wrong_evaluation_schema(tmp_path):
    suite=tmp_path/'suite.json';corpus=tmp_path/'data.jsonl'
    corpus.write_text(json.dumps({'messages':[{'role':'user','content':'Hello.'}]}))
    for value in ([],[{'id':'hidden','messages':[{'role':'user','content':'Hello.'}]}]):
        suite.write_text(json.dumps(value))
        with pytest.raises(ValueError):checker.audit([suite],[corpus])
