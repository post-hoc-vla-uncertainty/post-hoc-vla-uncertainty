"""Shared, offline architecture guide for the two pinned SmolVLA studies.

Presentation only: reads declared recipes, never loads a policy or changes a run.
Block numbers here are one-based; extraction code uses zero-based indices.
"""
import base64
from collections import defaultdict
from html import escape
import json


# Verified against the pinned policy config, SmolVLM2 config (including resolved
# vision defaults), both FeatureExtractor implementations, and the expert model.
# Keep this specific to this experiment, rather than claiming all SmolVLAs have
# this architecture. Dimensions below precede GP compression/coordinate selection.
LOCATIONS = {
    'visual_local': ('V1', 'Early vision / local visual context', '24,576',
        'After vision block 1, before the other 11 vision blocks. Pool each camera to a 4 × 4 grid: 2 × 16 × 768.'),
    'visual_patches': ('V1', 'Visual patches', '768 per patch',
        'Same block-1 output, retaining the 16 × 16 native patch grid per camera. Score patches separately, then aggregate.'),
    'visual_scene': ('V2', 'Late vision / scene visual context', '6,144',
        'After all 12 vision blocks and the final vision LayerNorm, before the image connector. Pool each camera to 2 × 2: 2 × 4 × 768.'),
    'language': ('L0', 'Initial language', '7,681',
        'Instruction token lookup embeddings, before image–language attention. Eight ordered token pools × 960, plus token count.'),
    'state': ('S0', 'Raw robot state', '8',
        'Prepared, normalized robot-state coordinates before the learned 8 → 960 state projection. “Raw” means no learned embedding.'),
    'joint_early': ('C1', 'Early fused image–language context', '2,888',
        'After multimodal block 1. Mean of each camera’s tokens + mean of language tokens + 8 raw state coordinates: 3 × 960 + 8.'),
    'joint_intermediate': ('C16', 'Intermediate fused context', '2,888',
        'After multimodal block 16 (code index 15), before block 17. Same three token-group means + raw state as C1.'),
    'contextual_state_intermediate': ('S16', 'Intermediate fused / contextualized state', '960',
        'Only the state token after multimodal block 16. This token has attended to images, language, and robot state. It is not the 2,888-value pooled context.'),
    'joint_late': ('C32', 'Final / late fused context', '2,888',
        'After all 32 multimodal blocks and final RMSNorm. Mean of each camera’s tokens + mean of language tokens + raw state.'),
    'contextual_language': ('L32', 'Contextual language', '7,681',
        'Language-token positions after all 32 multimodal blocks and final RMSNorm. Eight ordered pools × 960 + token count; now image-conditioned.'),
    'contextual_vision': ('V32', 'Contextual vision', '1,920',
        'Image-token positions after all 32 multimodal blocks and final RMSNorm. One mean per camera: 2 × 960; now language-conditioned.'),
    'contextual_state_late': ('S32', 'Final fused / contextualized state', '960',
        'Only the state token after all 32 multimodal blocks and final RMSNorm. Contains image, language, and state information.'),
    'action_context': ('C32', 'Context + noisy action', '3,239',
        'C32 (2,888) concatenated with current noisy action chunk xₜ (50 × 7) and scalar flow time t. This is a constructed GP input, not another transformer layer.'),
    'joint_early_flow': ('C1', 'Early context → decoder input', '3,239',
        'C1 (2,888) + current noisy action chunk (350) + flow time (1). Follow-up mapping input; the noise and time condition the prediction.'),
    'joint_intermediate_flow': ('C16', 'Intermediate context → decoder input', '3,239',
        'C16 (2,888) + current noisy action chunk (350) + flow time (1). Follow-up mapping input.'),
    'action_expert_3': ('D29', 'Decoder three blocks before output', '27,239',
        'After action-decoder block 29 (code index 28), before blocks 30–32 and final RMSNorm. Flatten 50 × 480 features; append xₜ, t, and C32.'),
    'action_expert_1': ('D31', 'Decoder one block before output', '27,239',
        'After action-decoder block 31 (code index 30), before block 32 and final RMSNorm. Flatten 50 × 480 features; append xₜ, t, and C32.'),
    'action_expert': ('D32', 'Final decoder / before output projection', '27,239',
        'After all 32 action-decoder blocks and final RMSNorm, immediately before the velocity projection. Flatten 50 × 480 = 24,000; append xₜ (350), t (1), and C32 (2,888).'),
    'decoder_hidden': ('D32', 'Final decoder embedding as KV-T target', '24,000',
        'Same physical tap as final decoder, but only the 50 × 480 hidden values. The target excludes appended context, noise, and time.'),
    'velocity': ('F', 'Action-chunk velocity field', '350',
        'Linear projection 480 → 7 at each of 50 action positions. One velocity vector for the current xₜ and t; not yet the completed action chunk.'),
    'action_chunk': ('A', 'Final generated action chunk', '350',
        '50 × 7 action values after all ten Euler integration steps. This is the old study’s final_action target.'),
    'vlm_logits': ('T', 'Language-model token logits', 'Vocabulary-sized per token',
        'The VLM language-model head applied to prefix representations. Used by token entropy/perplexity reference scores, not to generate robot actions.'),
}


def head_locations(recipe):
    """Resolve the actual feature bank, not a possibly historical group label."""
    method = recipe.get('method', '')
    features = recipe.get('features', {})
    if recipe.get('scope') == 'paper' or recipe.get('group') == 'paper':
        bank = ('vlm_logits' if method.startswith('vlm_') else
                'velocity' if method == 'vfd' else 'action_chunk')
        return dict(source=bank, target=None, conditioning=[], paper=True)
    bank = features.get('bank', '')
    if bank == 'action_expert' and features.get('expert_distance', 0):
        bank += '_' + str(features['expert_distance'])
    target = features.get('target_bank') if method == 'kv_t' else None
    flow = recipe.get('scope') == 'action'
    conditioning = []
    if flow:
        conditioning.append('Z' if target == 'action_chunk' else 'N')
        if bank.startswith('action_expert') and features.get('expert_context', True):
            conditioning.append('C32')
    return dict(source=bank, target=target, conditioning=conditioning, paper=False)


def comparison_locations(recipes):
    """Collapse compression/seed versions, preserving every distinct mapping."""
    groups = defaultdict(list)
    unique = {r.get('id', str(i)): r for i, r in enumerate(recipes)}
    for recipe in unique.values():
        pos = head_locations(recipe)
        key = (recipe.get('method', ''), recipe.get('group', ''), pos['source'],
               pos['target'], tuple(pos['conditioning']), pos['paper'])
        groups[key].append(recipe.get('id', 'unnamed'))
    rows = []
    for (method, group, source, target, conditioning, paper), names in sorted(groups.items(), key=lambda item: str(item[0])):
        source_info = LOCATIONS.get(source, ('?', source or 'Unspecified input', '—', 'No mapped location in the saved recipe.'))
        target_info = LOCATIONS.get(target) if target else None
        title = (method.upper().replace('_', '-') + ' · ' + source_info[1] +
                 (' → ' + target_info[1] if target_info else ''))
        codes = [source_info[0]]
        description = source_info[3]
        if conditioning and 'N' in conditioning:
            description += ' The GP input also includes current xₜ and t.'
        if 'Z' in conditioning:
            description += ' Here the GP input uses the initial Gaussian chunk z, rather than current xₜ, plus t at the final selected flow step.'
        if target_info:
            description += ' KV-T target: ' + target_info[3]
        if paper:
            description += ' Reference score from model predictions; no extra GP is attached.'
        rows.append(dict(title=title, group=group, method=method, source=source,
                         target=target, codes=codes, targets=[target_info[0]] if target_info else [],
                         conditioning=list(conditioning), description=description,
                         count=len(names), recipes=names))
    return rows


SVG_CSS = """
.mm-architecture text{font-family:Arial,sans-serif;fill:#18334b}
.mm-architecture .title{font-size:22px;font-weight:bold}
.mm-architecture .label{font-size:17px;font-weight:bold}.mm-architecture .detail{font-size:15px;fill:#415a6b}
.mm-architecture .lane{fill:#f4f7fa;stroke:#d7e1e9;stroke-width:1.5}
.mm-architecture .box{fill:white;stroke:#a2b4c3;stroke-width:1.5}
.mm-architecture .wire{fill:none;stroke:#637c90;stroke-width:2;marker-end:url(#mm-arrow)}
.mm-architecture .cache{fill:none;stroke:#9b7c45;stroke-width:2;stroke-dasharray:6 4;marker-end:url(#mm-cache-arrow)}
.mm-architecture .tap rect{fill:#eff6ff;stroke:#2164ae;stroke-width:2}.mm-architecture .tap text{fill:#174f90;font-size:14px;font-weight:bold}
.mm-architecture .tap.inactive rect{fill:#f4f5f6;stroke:#b7bfc6;stroke-dasharray:4 3}.mm-architecture .tap.inactive text{fill:#747f89}
.mm-architecture .tap.source rect{fill:#dbeafe;stroke:#195cb5;stroke-width:3}.mm-architecture .tap.target rect{fill:#ede2ff;stroke:#8247b2;stroke-width:3}
.mm-architecture .tap.conditioning rect{fill:#fff1ce;stroke:#a56b0a;stroke-width:3}
.mm-architecture .tap.target text{fill:#672995}.mm-architecture .tap.conditioning text{fill:#855000}
"""


def model_svg(active_codes=()):
    """All repeated blocks are accounted for; taps are branches off this path."""
    active_codes = set(active_codes)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" class="mm-architecture" viewBox="0 0 1280 1840" role="img" aria-labelledby="mm-title mm-desc">',
        '<title id="mm-title">SmolVLA architecture and uncertainty feature locations</title>',
        '<desc id="mm-desc">Two cameras share a 12-block vision encoder. Image, language, and state tokens pass through 32 multimodal blocks. Their per-layer attention cache conditions a 32-block action decoder, reused over ten flow steps. Labelled feature taps correspond to the glossary.</desc>',
        '<defs><marker id="mm-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#637c90"/></marker>',
        '<marker id="mm-cache-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#9b7c45"/></marker></defs>',
        '<style>'+SVG_CSS+'</style><rect width="1280" height="1840" fill="white"/>']
    def text(x, y, value, css='detail'):
        parts.append(f'<text x="{x}" y="{y}" class="{css}">{escape(value)}</text>')
    def box(x, y, width, height, title, *lines):
        parts.append(f'<rect class="box" x="{x}" y="{y}" width="{width}" height="{height}" rx="7"/>')
        text(x+14, y+25, title, 'label')
        for i, line in enumerate(lines): text(x+14, y+47+20*i, line)
    def wire(path, cache=False): parts.append(f'<path class="{"cache" if cache else "wire"}" d="{path}"/>')
    def tap(x, y, codes):
        for code in codes:
            width = max(36, 15+len(code)*10)
            cls = 'tap' + ('' if code in active_codes else ' inactive')
            parts.append(f'<g class="{cls}" data-site="{code}"><rect x="{x}" y="{y}" width="{width}" height="25" rx="5"/>'
                         f'<text x="{x+8}" y="{y+18}">{code}</text></g>')
            x += width+7

    text(28, 31, '1 · Encode the observation (before action-generation noise)', 'title')
    for x, w in [(24, 392), (444, 392), (864, 392)]:
        parts.append(f'<rect class="lane" x="{x}" y="48" width="{w}" height="556" rx="10"/>')
    box(40, 64, 360, 65, 'Two RGB cameras', 'Workspace + wrist · each 256 × 256')
    wire('M220 129 V151')
    box(40, 152, 360, 65, 'Patch embedding + positions', '16 × 16 pixels/patch → 256 tokens × 768')
    wire('M220 217 V239')
    box(40, 240, 360, 65, 'Vision block 1', 'Width 768 · shared camera encoder')
    tap(335, 274, ['V1'])
    wire('M220 305 V335')
    box(40, 336, 360, 65, 'Vision blocks 2–12 · 11×', 'LayerNorm → attention; LayerNorm → MLP')
    wire('M220 401 V427')
    box(40, 428, 360, 45, 'Final vision LayerNorm')
    tap(335, 438, ['V2'])
    wire('M220 473 V509')
    box(40, 510, 360, 76, 'Image connector', 'Group 4 × 4 patches; linear 12,288 → 960', '16 image tokens per camera')
    box(460, 64, 360, 65, 'Language instruction', 'Tokenizer → instruction token IDs')
    wire('M640 129 V151')
    box(460, 152, 360, 65, 'Token embedding lookup', '960 values per token')
    tap(755, 185, ['L0'])
    box(880, 64, 360, 65, 'Robot state', '8 prepared / normalized coordinates')
    tap(1175, 97, ['S0'])
    wire('M1060 129 V151')
    box(880, 152, 360, 65, 'State projection', 'Linear 8 → 960; one state token')
    text(476, 310, 'No image context at L0.')
    text(476, 334, 'Image–language attention starts below.')
    text(896, 310, 'S0 is the coordinate vector.')
    text(896, 334, 'S16 / S32 are learned state tokens.')
    # The three paths merge once into the observation prefix.
    wire('M220 586 V623 H296 V655')
    wire('M640 217 V245 H812 V605 H480 V655')
    wire('M1060 217 V245 H1238 V623 H564 V655')
    box(32, 656, 568, 65, '2 · Multimodal transformer · observation prefix', 'Concatenate 32 image tokens + instruction + one state token')
    parts.append('<rect class="lane" x="24" y="740" width="584" height="538" rx="10"/>')
    wire('M306 721 V757')
    box(40, 758, 552, 66, 'Multimodal block 1', 'Token width 960; first image–language fusion')
    tap(526, 792, ['C1'])
    wire('M306 824 V871')
    box(40, 872, 552, 66, 'Multimodal blocks 2–16 · 15×', 'Midpoint: after block 16, before block 17')
    tap(462, 906, ['C16', 'S16'])
    wire('M306 938 V985')
    box(40, 986, 552, 66, 'Multimodal blocks 17–32 · 16×', 'Each block: RMSNorm → attention; RMSNorm → MLP')
    wire('M306 1052 V1103')
    box(40, 1104, 552, 90, 'Final multimodal RMSNorm', 'Same output tensor; heads select different token groups')
    tap(307, 1165, ['C32', 'L32', 'V32', 'S32'])
    text(54, 1225, 'C = pooled camera + language tokens, with raw state appended.')
    text(54, 1251, 'L / V / S = select only language / image / state-token positions.')
    wire('M40 1140 H14 V1330 H38')
    box(40, 1298, 552, 66, 'Optional VLM language-model output head', 'Linear 960 → vocabulary logits; token reference scores')
    tap(533, 1332, ['T'])
    box(40, 1400, 552, 119, 'What can attend to what?',
        'Image and language tokens attend to each other, not state.',
        'The state token attends to images, language, and state.',
        'Prefix tokens never attend to noisy action tokens.',
        'The decoder reads the full per-layer cache, not C32 pooling.')
    box(40, 1545, 552, 137, 'Inside each repeated transformer block',
        'Norm → Q/K/V projections → attention → output projection',
        '→ residual add → norm → MLP → residual add.',
        'Vision: width 768, MLP 3,072, LayerNorm.',
        'Multimodal: width 960, gated MLP 2,560, RMSNorm.',
        'Action decoder: width 480, gated MLP 1,280, RMSNorm.')

    text(702, 641, '3 · Generate the action chunk', 'title')
    parts.append('<rect class="lane" x="694" y="656" width="558" height="1061" rx="10"/>')
    box(710, 672, 526, 86, 'Initial Gaussian z → current xₜ',
        '50 actions × 7 coordinates; start at t = 1', 'Current xₜ and time t change at each integration step')
    tap(1120, 683, ['Z', 'N'])
    wire('M973 758 V791')
    box(710, 792, 526, 65, 'Action projection + sinusoidal time encoding',
        'Per action: 7 → 480; time t → 480; concatenate → 960')
    wire('M973 857 V879')
    box(710, 880, 526, 65, 'Action–time MLP', 'Linear 960 → 480 → SiLU → linear 480 → 480')
    wire('M973 945 V975')
    box(710, 976, 526, 66, 'Decoder blocks 1–28 · 14× pair',
        'Odd block: self/prefix attention; even block: cross-attention')
    wire('M973 1042 V1067')
    box(710, 1068, 526, 48, 'Decoder block 29 · self/prefix attention')
    tap(1170, 1079, ['D29'])
    wire('M973 1116 V1143')
    box(710, 1144, 526, 48, 'Decoder block 30 · cross-attention')
    wire('M973 1192 V1219')
    box(710, 1220, 526, 48, 'Decoder block 31 · self/prefix attention')
    tap(1170, 1231, ['D31'])
    wire('M973 1268 V1295')
    box(710, 1296, 526, 48, 'Decoder block 32 · cross-attention')
    wire('M973 1344 V1371')
    box(710, 1372, 526, 65, 'Final decoder RMSNorm', '50 × 480 values, already conditioned on observation, xₜ, t')
    tap(1170, 1384, ['D32'])
    wire('M973 1437 V1463')
    box(710, 1464, 526, 65, 'Velocity output projection · linear 480 → 7', 'v(xₜ, t, observation): 50 × 7 values')
    tap(1175, 1498, ['F'])
    wire('M973 1529 V1555')
    box(710, 1556, 526, 65, 'Euler step: x ← x − v / 10; t ← t − 0.1', '10 steps in total, reusing the SAME 32 decoder blocks')
    wire('M1236 1586 H1265 V745 H1236')
    wire('M973 1621 V1647')
    box(710, 1648, 526, 65, 'Final action chunk', '50 × 7; execute the first 25 actions, then replan')
    tap(1175, 1681, ['A'])
    # Layer-aligned cache bus: avoid implying that pooled C32 drives the decoder.
    parts.append('<path d="M592 785 H618 V1019 H592 M592 905 H618 M618 940 H660 V1320" fill="none" stroke="#9b7c45" stroke-width="2" stroke-dasharray="6 4"/>')
    for y in (1008, 1092, 1168, 1244, 1320): wire(f'M660 {y} H708', cache=True)
    parts.append('<text x="646" y="1290" class="detail" transform="rotate(-90 646 1290)">Per-layer attention keys / values · 32 caches</text>')
    text(28, 1753, 'Blue labelled taps: feature sources or targets declared in this study. Grey dashed taps: architecture locations not used here.')
    text(28, 1779, 'Head branches: extract / pool features → select or project dimensions → optional whitening → GP variance → uncertainty score.')
    text(28, 1805, 'Heads read a frozen policy; they do not alter its actions. Block numbers are 1-based; code indices are one smaller.')
    parts.append('</svg>')
    return ''.join(parts)


HTML_CSS = """
.model-map{color:#18334b;font:15px/1.55 system-ui,sans-serif;margin:24px 0}
.model-map h2{margin:0 0 12px}.model-map p{max-width:1200px}.model-map .mm-scroll{overflow-x:auto}
.model-map svg{display:block;width:100%;min-width:760px;max-width:1280px;margin:auto}
.model-map label{font-weight:600;display:block}.model-map select{display:block;width:100%;padding:10px;font:inherit;margin:8px 0;border:1px solid #9aafbf;border-radius:6px}
.model-map .mm-controls{padding:16px;background:#f2f6fa;border-radius:8px;margin:16px 0}
.model-map .mm-description{min-height:3em;margin:10px 0}.model-map .mm-key{font-size:13px}
.model-map .mm-source{color:#195cb5;font-weight:700}.model-map .mm-target{color:#8247b2;font-weight:700}.model-map .mm-conditioning{color:#986208;font-weight:700}
.model-map table{border-collapse:collapse;width:100%;min-width:720px;font-size:13px;line-height:1.5}
.model-map td,.model-map th{text-align:left;padding:10px;border:1px solid #dbe4eb;vertical-align:top}
.model-map th{background:#eef4f9}.model-map td code{display:inline;min-width:0;font-size:12px;overflow-wrap:anywhere}
.model-map .mm-selected td{background:#eaf3ff}.model-map .mm-muted{color:#697987}
.model-map button{font:inherit;color:#195cb5;background:white;border:1px solid #b9cbdc;border-radius:4px;padding:4px 8px;cursor:pointer}
.model-map summary{cursor:pointer;font-weight:600;margin:16px 0}.model-map .mm-note{border-left:3px solid #a6bcd1;padding-left:15px}
@media print{.model-map .mm-controls{display:none}.model-map svg{min-width:0}.model-map{break-inside:auto}.model-map tr{break-inside:avoid}}
"""


def model_map_html(recipes, *, study):
    if study not in ('old', 'followup'): raise ValueError('Unknown model-map study')
    rows = comparison_locations(recipes)
    used = defaultdict(set)
    for row in rows:
        used[row['source']].add('Reference score' if row['group'] == 'paper' else row['method'].upper().replace('_', '-')+' input')
        if row['target']: used[row['target']].add('KV-T target')
    # Aliases share their physical observation tap but carry flow conditioning.
    for alias, bank in [('joint_early_flow','joint_early'), ('joint_intermediate_flow','joint_intermediate'), ('action_context','joint_late')]:
        if used[alias]: used[bank].add('Part of action-conditioned input')
    active = {code for row in rows for key in ('codes','targets','conditioning') for code in row[key]}
    svg = model_svg(active)
    encoded_svg = base64.b64encode(svg.encode()).decode()
    options = '<option value="">All declared locations</option>' + ''.join(
        f'<option value="{i}">{escape(r["title"])} ({r["count"]} recipes)</option>' for i,r in enumerate(rows))
    legend = []
    for bank, (code, name, dimensions, explanation) in LOCATIONS.items():
        roles = ', '.join(sorted(used[bank])) or 'Not used in this study'
        legend.append(f'<tr data-bank="{bank}"><td><b>{code}</b></td><td>{escape(name)}<br><code>{bank}</code></td>'
            f'<td>{dimensions}</td><td>{escape(explanation)}</td><td>{escape(roles)}</td></tr>')
    mappings = []
    for i, row in enumerate(rows):
        source = LOCATIONS.get(row['source'], ('?', row['source']))
        target = LOCATIONS.get(row['target'])
        path = source[0] + (' → '+target[0] if target else '')
        if row['conditioning']: path += '; add '+', '.join(row['conditioning'])
        mappings.append(f'<tr><td><button type="button" data-mapping="{i}">Show {escape(path)}</button></td>'
            f'<td>{escape(row["method"].upper().replace("_","-"))}</td><td><code>{escape(row["group"])}</code></td>'
            f'<td>{escape(row["title"])}</td><td>{row["count"]}</td></tr>')
    note = ('The old study retains all its original locations, including visual patches, raw state, and earlier decoder blocks.' if study == 'old' else
        'The follow-up retains the observation KV heads and four central KV/KV-T mappings: C1 → D32, C16 → D32, C16 → C32, and D32 → F. '
        'Visual-patch GPs, raw-state GPs, and D29/D31 heads are absent from this study; their model locations remain visible in grey where unused.')
    followup = ('' if study == 'old' else
        '<p>For C1/C16 → D32 in the current follow-up, compress/select the 2,880 image–language values and then append all 359 conditioning values '
        '(state 8 + xₜ 350 + t 1). Thus a 512-dimensional projection gives up to 871 GP inputs. D32 targets contain only the 24,000 hidden values, '
        'subsequently projected to 16 output coordinates.</p>')
    payload = json.dumps(rows, ensure_ascii=True).replace('<','\\u003c')
    return ('<section id="model-map" class="model-map"><style>'+HTML_CSS+'</style>'
        '<h2>Model architecture and uncertainty-head locations</h2>'
        '<p>This is the <b>actual pinned SmolVLA used in these studies</b>: a 12-block vision encoder, a 32-block multimodal transformer, '
        'and a 32-block action decoder. Numbered badges mark extraction points; the table below translates every feature-bank name. '
        'Widths in the diagram are model widths, before GP preprocessing.</p><p>'+note+'</p>'
        '<div class="mm-controls"><label for="mm-choice">Highlight a head location or KV-T input → target mapping</label>'
        '<select id="mm-choice">'+options+'</select><p class="mm-key"><span class="mm-source">Blue = input/source</span> · '
        '<span class="mm-target">Purple = KV-T target</span> · <span class="mm-conditioning">Gold = appended conditioning</span>. '
        'N = current noisy action chunk + time; Z = initial Gaussian chunk.</p>'
        '<p class="mm-description" aria-live="polite">Select a mapping to see exactly where its input and target come from. '
        'Compression, whitening, and random-seed variants at the same location are grouped together.</p></div>'
        '<div class="mm-scroll">'+svg+'</div><p><a download="smolvla-head-locations-'+study+'.svg" href="data:image/svg+xml;base64,'+encoded_svg+'">Download schematic (SVG)</a></p>'
        '<div class="mm-note"><p><b>“Fused context” and “fused state” are different.</b> Context C1/C16/C32 means three pooled token groups '
        '(two cameras and language) with raw state appended. State S16/S32 means the single learned state token after attention. '
        'Image/language tokens cannot attend to state in this policy; the state token can attend to all three modalities.</p>'
        '<p><b>“Final context” is not “final decoder.”</b> C32 depends only on the observation. D32 is inside action generation and already depends on xₜ and t. '
        'Noise enters <em>before</em> the decoder. The velocity projection after D32 is linear; finishing the action chunk requires ten integration steps.</p>'
        '<p><b>The heads are readout branches.</b> Pooling/flattening and random coordinates, PCA, PLS, or whitening occur outside the policy. '
        'KV scores input-space GP variance. KV-T additionally fits a mapping to the stated target, usually compressed to 16 outputs. '
        'KV with PLS still uses targets to construct its input projection. “KV” here means kernel variance; the model’s attention key/value cache is a different use of the letters.</p></div>'
        '<h3>Location legend: names, exact positions, and feature sizes</h3>'
        '<p>Sizes below are before coordinate selection / PCA / PLS. Action-conditioned modality inputs add xₜ (350) and t (1) to the listed observation feature. '
        'Action probes use the initial robot observation, without stepping the simulator; these recipes sample flow steps 0, 5, and 9 (t ≈ 1, 0.5, 0.1). '
        'They use four Gaussian probes per support observation and 50 per scoring observation. The final-action mapping uses its declared final flow step.</p>'
        +followup+'<div class="mm-scroll"><table><thead><tr><th>Tap</th><th>Report name / feature bank</th><th>Raw feature size</th><th>What is extracted</th><th>Declared use</th></tr></thead><tbody>'
        +''.join(legend)+'</tbody></table></div><details><summary>All declared head locations and mappings (grouped across parameter variants)</summary>'
        '<p>These are declared configurations, including unfinished or infeasible ones; membership here does not claim a successful result. '
        'Paper/reference methods use token logits, action samples, or velocity comparisons rather than GP feature taps. '
        'The decoder alternates attention to prefix + action tokens and cross-attention to the prefix alone.</p>'
        '<div class="mm-scroll"><table><thead><tr><th>Locate</th><th>Estimator</th><th>Configuration group</th><th>Input → target</th><th>Recipes</th></tr></thead><tbody>'
        +''.join(mappings)+'</tbody></table></div></details>'
        '<details><summary>Architecture sources and numbering</summary><p>Checked against '
        '<code>storage/outputs/pretrain/libero_pretrain_leak3/seed_0/checkpoints/030000/pretrained_model/config.json</code>, '
        'its pinned SmolVLM2-500M-Video-Instruct configuration, '
        '<code>upstream/uq_vla_extensions/src/lerobot/policies/smolvla/{modeling_smolvla,smolvlm_with_expert}.py</code>, '
        '<code>src/uq_extensions/features.py</code>, and <code>src/uq_followup/features.py</code>. '
        'The diagram uses human block numbers 1–32; code indices are 0–31. Intermediate context is code index 15. '
        'Earlier decoder taps are after code indices 28 and 30; final decoder is after the final RMSNorm. '
        'The diagram documents these pinned studies, not every SmolVLA configuration.</p></details>'
        '<script type="application/json" class="mm-data">'+payload+'</script><script>'+JS+'</script></section>')


JS = r"""
(() => {
  const root = document.getElementById('model-map');
  const rows = JSON.parse(root.querySelector('.mm-data').textContent);
  const choice = root.querySelector('#mm-choice');
  const description = root.querySelector('.mm-description');
  const initialDescription = description.textContent;
  function highlight() {
    const selected = choice.value === '' ? null : rows[Number(choice.value)];
    root.querySelectorAll('[data-site]').forEach(tap => {
      const code = tap.dataset.site;
      for (const role of ['source', 'target', 'conditioning']) tap.classList.remove(role);
      if (selected) {
        if (selected.conditioning.includes(code)) tap.classList.add('conditioning');
        if (selected.codes.includes(code)) tap.classList.add('source');
        if (selected.targets.includes(code)) tap.classList.add('target');
      }
    });
    root.querySelectorAll('[data-bank]').forEach(row => row.classList.toggle('mm-selected',
      !!selected && [selected.source, selected.target].includes(row.dataset.bank)));
    description.textContent = selected ? selected.description : initialDescription;
  }
  choice.addEventListener('change', highlight);
  root.querySelectorAll('[data-mapping]').forEach(button => button.addEventListener('click', () => {
    choice.value = button.dataset.mapping; highlight();
    root.scrollIntoView({behavior: 'smooth', block: 'start'});
  }));
})();
"""
