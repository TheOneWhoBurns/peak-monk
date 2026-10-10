"""Author-reviewed positive on-policy replies; only final reply receives loss."""
import hashlib,json
from pathlib import Path
from collections import Counter

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
selected={
'lantern_s73':{4:'Discusses particular objects and memories without inventing caller facts.',6:'Relevant question about whether Belen now knows about the dent.',11:'Relevant anticipation of Belen enjoying the visit.',12:'Responds to the drawer joke and asks about its actual fault.',17:'Brief coherent joke about leaving without shoes.',20:'Recognizes the short successful negotiation.',24:'Keeps using the bowl for breakfast rather than inventing an obligation.',29:'Continues the comic floor accusation.',31:'Accepts the caller decision to stop expanding the job.',32:'Brief personification consistent with the caller closing the day.'},
'library_s73':{3:'Relevant fern and food joke.',4:'Answers quiet-afternoon preference in his own life.',5:'Asks a relevant choice about the remaining book sorting.',7:'Appreciates informal display without a grand opening.',8:'Expresses ordinary preference about rain without controlling Neri.',11:'Relevant tin-storage response.',20:'Responds to the cafe-size joke with curiosity.',23:'Fallible personal experience of rehearsing conversation.',25:'Responds to folding the map without returning to advice.',26:'Allows imperfect comfort with silence rather than claiming enlightenment.',31:'Accepts the simple bread and soup plan.',32:'Acknowledges enjoying ordinary conversation.'},
'lantern_s89':{6:'Gentle joke about blaming weather, no caller-fact invention.',8:'Accepts a pause for a sandwich.',12:'Brief continuation of the anthropomorphic drawer joke.',14:'Recognizes both visit and walk as reasons to go.',15:'Ordinary preference for eating over planning.',18:'Coherent joke about relatives arguing about a route.',19:'Gentle comparison of a small disagreement with the lantern.',22:'Asks how Belen feels before imposing an interpretation.',26:'Enjoys the image of carrying toast.',29:'Continues the floor accusation joke.',31:'Accepts stopping and expresses a similar ordinary preference.',32:'Brief completion acknowledgment.'},
'library_s89':{1:'Relevant question about the returned book.',3:'Connects fern bookmark to botanical collection and book preservation.',5:'Responds to the waiting terrier with humor and a question.',7:'Brief joke about separating biscuits from found bookmarks.',8:'Acknowledges company and rain.',10:'Recognizes comic triumph over the misplaced umbrella.',14:'Ordinary interest in sandwich memory.',19:'Personal preference for shared extra chips.',23:'Recognizes ordinary conversation as relief from performance.',25:'Dry reply to watching the map folding.',31:'Accepts the caller simple evening plan.'}}
rows=[];reviews=[]
def add(path,r,exchange,reason):
 messages=r['messages'][:exchange*2];assert len(messages)==exchange*2
 ident=f"retention_{r['id']}_{exchange:02d}"
 rows.append(dict(id=ident,family=r['family'],root_id=r['root_id'],split='train',messages=messages,supervised_message_indices=[len(messages)-1]))
 reviews.append(dict(id=ident,path=str(path),source_sha256=sha(path),exchange=exchange,accepted_reply=messages[-1]['content'],reason=reason))
for root,exchanges in selected.items():
 path=Path(f'results/dialogue-pilot-05/{root}_spine32.json');r=json.loads(path.read_text())
 for exchange,reason in exchanges.items():add(path,r,exchange,reason)
branches={
'lantern_s73_mentor32':{33:'Explicit Mara and she; practical characterization without supernatural claims.'},
'library_s73_mentor32':{33:'Mara and she with coherent attention practice.',34:'Hypothetical reaction by deceased teacher, maintaining she and ordinary attention.'},
'library_s73_challenge32':{33:'Explicit living man identity in the fiction, grounded in own kettle setting.'},
'lantern_s89_challenge32':{33:'Declines warranty-service role and returns to the conversation.'},
'library_s89_challenge32':{33:'Declines becoming the automated library service without assistant disclosure.'}}
for ident,exchanges in branches.items():
 path=Path(f'results/dialogue-pilot-06/{ident}.json');r=json.loads(path.read_text())
 for exchange,reason in exchanges.items():add(path,r,exchange,reason)
out=Path('data/student-retention-v1');out.mkdir(exist_ok=False)
(out/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
report=dict(status='Author-reviewed training candidates; not yet used for an update',reviewer='Codex author, not independent',examples=len(rows),families=dict(Counter(r['family'] for r in rows)),root_counts=dict(Counter(r['root_id'] for r in rows)),data_sha256=sha(out/'train.jsonl'),objective='Only the selected final sampled reply receives loss. Earlier uncorrected student output is context, never a positive target.',scope='Positive continuations selected from the already-exposed training-only student paths. Not a new independent dataset or evidence of generalization. Intended retention alternative to nearlyzero-loss canonical replay.',reviews=reviews)
(out/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='reviews'}))
