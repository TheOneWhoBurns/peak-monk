import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_gate import gate


def test_overall_score_cannot_hide_identity_failure(tmp_path):
    cases=[dict(id=f'd{i}',category='identity_stability' if i==0 else 'ordinary',turns=['test']) for i in range(10)]
    suite=tmp_path/'roleplay-v2.json';suite.write_text(json.dumps(cases))
    sha=hashlib.sha256(suite.read_bytes()).hexdigest()
    transcript=dict(suite=str(suite),suite_sha256=sha,validation_loss=1.,conversations=cases)
    review=dict(reviewer='unit-test fixture',suite_sha256=sha,ratings=[dict(id=c['id'],acceptable=i!=0,severe_failure=False,reason='fixture') for i,c in enumerate(cases)])
    result=gate(transcript,review,2.)
    assert result['acceptable_fraction']==.9
    assert not result['pass']
    assert result['required_character_failures']==['d0']
    review['ratings'][0]['acceptable']=True
    assert gate(transcript,review,2.)['pass']


def test_frozen_final_without_categories_still_requires_character_control():
    suite=Path(__file__).resolve().parents[1]/'evaluation/final-roleplay-v2.json'
    cases=json.loads(suite.read_text())
    sha=hashlib.sha256(suite.read_bytes()).hexdigest()
    transcript=dict(suite=str(suite),suite_sha256=sha,validation_loss=1.,conversations=cases)
    review=dict(reviewer='unit-test fixture',suite_sha256=sha,ratings=[
        dict(id=c['id'],acceptable=c['id']!='f04',severe_failure=False,reason='fixture') for c in cases])
    result=gate(transcript,review,2.)
    assert result['acceptable_count']==14
    assert not result['pass']
    assert result['required_character_failures']==['f04']
