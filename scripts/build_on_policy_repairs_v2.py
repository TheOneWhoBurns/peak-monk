"""Add reviewed coherence repairs from actual student histories, not fabricated rollouts."""
import hashlib
import json
from pathlib import Path
from collections import Counter


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# One-based exchanges. Earlier sampled replies remain unchanged and unsupervised.
selected = {
    'lantern_s73': {
        2: ('It has already earned its place, then. A lantern can be useful even when the fishing is hopeless.',
            'Asks what Belen used the lantern for immediately after the caller says she took it fishing.'),
        9: ('I have bought those. A little salt can help, but it cannot supply an entirely missing flavour.',
            'Introduces onions and asks about eating them although the caller described cheese and tomato.'),
        19: ('You want a later start; Tomas wants to get going. Who sends the next small message?',
             'Changes a departure-time negotiation into a question of which person will get an invented picnic.'),
        23: ('It sounds as if she wanted the bowl used, rather than given a place of honour. Did using it get easier?',
             'Asserts the caller did not tell Belen about using the bowl, which the history does not establish.'),
    },
    'lantern_s89': {
        4: ('Both. My chipped blue cup is useful, but I am also fond of it. Another cup would hold tea just as well.',
            'Evades an ordinary question about his own possessions by talking incoherently about the caller\'s dead moth.'),
        20: ('Then eight it is. I like a negotiation that ends before the visit begins.',
             'Treats Tomas\'s acceptance as undecided even though Tomas offered eight and the caller accepted.'),
        28: ('I would move the mustard before the tools acquire an interesting coating.',
             'Restates the mustard as being on the tools\' side when the caller explicitly put them on opposite sides.'),
    },
    'library_s73': {
        27: ('I would enjoy some of those hours. A quiet room with something small to do suits me.',
             'Repeats the earlier forgotten-story/lunch worry instead of responding to hours of quiet at the library.'),
    },
    'library_s89': {
        12: ('Then your cookbook has its own little collection. What are the postcards of?',
             'Conflates the caller\'s postcards with Jun\'s earlier rook rather than addressing the discovered objects.'),
        21: ('That sounds like a lively table. Perhaps you should warn the cafe.',
             'Invents an unpleasant lunch from which Neri was saved, contrary to the caller\'s pleased account.'),
        28: ('It can stay there until one of you makes the soup. I leave scraps of paper in books for years.',
             'Says the postcards have become the recipe, confusing the bookmark with what it marks.'),
        29: ('I have come back from the village with something interesting and forgotten the ordinary thing I went for.',
             'Talks about postcards needing someone who has a recipe although the caller already identified that recipe and is discussing ingredients.'),
    },
}

out = Path('data/on-policy-repairs-v2')
out.mkdir(exist_ok=False)
prior = Path('data/on-policy-repairs-v1/train.jsonl')
rows = [json.loads(line) for line in prior.read_text().splitlines()]
reviews = []
for root, exchanges in selected.items():
    path = Path(f'results/dialogue-pilot-05/{root}_spine32.json')
    record = json.loads(path.read_text())
    assert record['split'] == 'train'
    for exchange, (replacement, reason) in exchanges.items():
        messages = [dict(message) for message in record['messages'][:exchange*2]]
        assert len(messages) == exchange*2 and messages[-1]['role'] == 'assistant'
        rejected = messages[-1]['content']
        messages[-1]['content'] = replacement
        ident = f'{root}_coherence_{exchange:02d}'
        source = dict(path=str(path),sha256=sha(path),exchange=exchange)
        rows.append(dict(id=ident,family=record['family'],root_id=root,split='train',
            review='author-approved',messages=messages,supervised_message_indices=[len(messages)-1],
            source=source,repair_reason=reason))
        reviews.append(dict(id=ident,source=source,rejected_reply=rejected,
            corrected_reply=replacement,reason=reason))
assert len({row['id'] for row in rows}) == len(rows)
(out/'train.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
manifest = dict(status='Author-reviewed training candidates; not trained',reviewer='Codex author; not independent',
    repairs=len(rows),new_repairs=len(reviews),prior_data=str(prior),prior_data_sha256=sha(prior),
    families=dict(Counter(row['family'] for row in rows)),roots=dict(Counter(row['root_id'] for row in rows)),
    data_sha256=sha(out/'train.jsonl'),objective='Only final corrected reply receives positive loss. All earlier student replies, including mistakes, remain unmodified context.',
    limitations='Still only two scenario families. These are targeted training repairs, not held-out examples or an estimate of generalization. Mild quirks and defensible alternative opinions were not automatically rejected.',
    teacher_lore='No new mentor-probe data yet. Add only after actual matched-depth generation and review, through a new version.',reviews=reviews)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='reviews'},indent=2))
