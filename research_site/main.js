function createTable(rows){
  return `<table>${rows.map(row=>`<tr>${row.map(cell=>`<td>${cell}</td>`).join('')}</tr>`).join('')}</table>`;
}

function formatSetSet(items){
  if(!items||items.length===0) return '<span class="muted">none</span>';
  return `<code>${items.slice(0,20).join(', ')}${items.length>20?' …':''}</code> <span class="muted">(${items.length} active)</span>`;
}

function scoreRow(label,value){
  const score = Number(value).toFixed(3);
  return `<div class="bar-row"><div>${label}</div><div>${score}</div></div><div class="bar-track"><div class="bar-fill" style="width:${Math.min(100,Math.max(0, value*100))}%"></div></div>`;
}

function renderV2V3(data){
  const episode = data.episode;
  const retrieval = data.retrieval_partial;
  const textRetrieval = data.retrieval_text;
  const html = `
    <p>This experiment illustrates pure neural retrieval and multimodal association from the notebook.</p>
    <div class="metric-grid">
      <div class="metric-card"><strong>Episode cue</strong><pre class="code-block">${JSON.stringify(episode.input, null, 2)}</pre></div>
      <div class="metric-card"><strong>Novelty and plasticity</strong>${scoreRow('Energy', episode.energy)}${scoreRow('Neuromodulator M', episode.M/2)}<p><strong>Familiar:</strong> ${episode.familiar}</p></div>
      <div class="metric-card"><strong>CA3 state</strong><p>${formatSetSet(episode.ca3_active)}</p></div>
    </div>
    <h3>Retrieval with partial SQL cue</h3>
    ${createTable([['Metric','Value'],['CA3 active', formatSetSet(retrieval.ca3_active)],['Relation prediction',retrieval.relation_prediction+' ('+retrieval.relation_confidence.toFixed(3)+')']])}
    <p><strong>Top SQL readout:</strong></p><pre class="code-block">${JSON.stringify(retrieval.sql_reconstruction, null, 2)}</pre>
    <p><strong>Top graph readout:</strong></p><pre class="code-block">${JSON.stringify(retrieval.graph_reconstruction, null, 2)}</pre>
    <h3>Retrieval from text cue</h3>
    <p><strong>Text cue:</strong> ${data.retrieval_text.text}</p>
    <p><strong>Predicted relation:</strong> ${data.retrieval_text.relation_prediction} (${data.retrieval_text.relation_confidence.toFixed(3)})</p>
    <p><strong>Completion curve:</strong></p><pre class="code-block">${JSON.stringify(data.completion_curve.curve, null, 2)}</pre>
  `;
  document.getElementById('v2v3-content').innerHTML = html;
}

function renderV4V5(data){
  const timeline = data.episodes.map(ep=>`<li><strong>Episode ${ep.episode_id}</strong> (${ep.text}) → predecessor ${ep.predecessor_episode_id}, successor ${ep.successor_episode_id}</li>`).join('');
  const metrics = data.metrics;
  const html = `
    <p>This experiment simulates episode ordering, continual memory, and consolidation metrics.</p>
    <h3>Encoded sequence</h3><ul>${timeline}</ul>
    <div class="metric-grid">
      <div class="metric-card"><strong>Interference</strong>${scoreRow('Mean retention', metrics.interference.mean_retention)}<p>Retention per episode: ${metrics.interference.retention.map(v=>v.toFixed(3)).join(', ')}</p></div>
      <div class="metric-card"><strong>Separation</strong>${scoreRow('Separation margin', metrics.separation.separation_margin_mean)}${scoreRow('Top-1 accuracy', metrics.separation.separation_top1_accuracy)}</div>
      <div class="metric-card"><strong>Continuity</strong>${scoreRow('Continuity margin', metrics.continuity.continuity_margin_mean)}${scoreRow('Link consistency', metrics.continuity.link_consistency)}</div>
    </div>
    <h3>Completion curve after consolidation</h3>
    <pre class="code-block">${JSON.stringify(metrics.completion_curve.curve, null, 2)}</pre>
  `;
  document.getElementById('v4v5-content').innerHTML = html;
}

function renderV6(data){
  const rows = [[ 'Use text', 'CA3 excitatory', 'Consolidate', 'Mean completion', 'Mean retention', 'Separation top1', 'Continuity link consistency' ]]
    .concat(data.benchmarks.map(row=>[
      row.use_text? 'yes':'no', row.ca3_exc, row.consolidate? 'yes':'no', row.mean_completion.toFixed(3), row.mean_retention.toFixed(3), row.separation_top1.toFixed(3), row.continuity_links.toFixed(3)
    ]));
  const html = `<p>Benchmark results compare configuration variants for the research harness.</p>${createTable(rows)}`;
  document.getElementById('v6-content').innerHTML = html;
}

function initNav(){
  document.querySelectorAll('header nav a').forEach(link=>{
    link.addEventListener('click', event=>{
      event.preventDefault();
      const target = document.querySelector(link.getAttribute('href'));
      if(target) target.scrollIntoView({behavior:'smooth', block:'start'});
    });
  });
}

window.addEventListener('DOMContentLoaded', ()=>{
  initNav();
  if(typeof SIM_DATA==='undefined'){document.getElementById('v2v3-content').innerText='Simulation data not loaded.';return}
  renderV2V3(SIM_DATA.v2v3);
  renderV4V5(SIM_DATA.v4v5);
  renderV6(SIM_DATA.v6);
});