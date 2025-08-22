const API_BASE = 'http://127.0.0.1:8000';

const $ = (sel) => document.querySelector(sel);
const input = $('#inputText');
const btn = $('#detectBtn');
const statusEl = $('#status');
const result = $('#result');
const humanPct = $('#humanPct');
const aiPct = $('#aiPct');
const labelEl = $('#label');

function setStatus(msg, isError=false) {
	statusEl.textContent = msg;
	statusEl.classList.toggle('error', isError);
}

function showResult(probHuman, probAI, label) {
	const human = Math.round(probHuman * 100);
	const ai = Math.round(probAI * 100);
	humanPct.textContent = human + '%';
	aiPct.textContent = ai + '%';
	labelEl.textContent = `Prediction: ${label}`;
	result.classList.remove('hidden');

	// adjust bar widths
	document.querySelector('.bar.human').style.width = human + '%';
	document.querySelector('.bar.ai').style.width = ai + '%';
}

btn.addEventListener('click', async () => {
	const text = (input.value || '').trim();
	if (!text) {
		setStatus('Please enter some text.', true);
		return;
	}

	setStatus('Detecting...');
	result.classList.add('hidden');

	try {
		const resp = await fetch(`${API_BASE}/predict`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ text })
		});
		if (!resp.ok) {
			const err = await resp.json().catch(() => ({}));
			throw new Error(err.detail || `Request failed: ${resp.status}`);
		}
		const data = await resp.json();
		showResult(data.prob_human, data.prob_ai, data.label);
		setStatus('');
	} catch (e) {
		console.error(e);
		setStatus(e.message || 'Error during detection', true);
	}
});
