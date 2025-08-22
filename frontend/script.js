const API_BASE = 'http://127.0.0.1:8000';

const $ = (sel) => document.querySelector(sel);
const input = $('#inputText');
const btn = $('#detectBtn');
const statusEl = $('#status');
const result = $('#result');
const humanPct = $('#humanPct');
const aiPct = $('#aiPct');
const labelEl = $('#label');
const wordCounter = $('#wordCounter');
const wordHelp = $('#wordHelp');
const MIN_WORDS = 30;
// Start with button disabled until minimum met
btn.disabled = true;

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

// Count committed words: a word counts only after a space, newline, or tab is typed after it
function countCommittedWords(text) {
	if (!text) return 0;
	const matches = text.match(/\b[A-Za-z0-9_]+(?=\s)/g) || [];
	return matches.length;
}

function updateWordState() {
	const committed = countCommittedWords(input.value);
	wordCounter.textContent = `${committed}/${MIN_WORDS} words`;
	const ok = committed >= MIN_WORDS;
	btn.disabled = !ok;
	btn.title = ok ? '' : `Enter at least ${MIN_WORDS} committed words (press space after each word)`;
	wordCounter.classList.toggle('met', ok);
	if (wordHelp) {
		if (ok) {
			wordHelp.textContent = 'Minimum reached.';
			wordHelp.classList.add('ok');
		} else {
			wordHelp.textContent = 'Please paste or enter at least 30 words.';
			wordHelp.classList.remove('ok');
		}
	}
	return committed;
}

// Multiple listeners to ensure update across environments
input.addEventListener('input', updateWordState); // fires on space/enter as well
input.addEventListener('change', updateWordState);

btn.addEventListener('click', async () => {
	const text = (input.value || '').trim();
	const words = countCommittedWords(text + ' '); // treat final word as committed on submit
	if (!text) {
		setStatus('Please enter some text.', true);
		return;
	}
	if (words < MIN_WORDS) {
		setStatus(`Please enter more than or equal to ${MIN_WORDS} words (currently ${words}).`, true);
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

// Initialize counter after a tiny delay to ensure DOM painted
window.requestAnimationFrame(updateWordState);
