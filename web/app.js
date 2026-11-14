const form = document.getElementById('request');
const model = document.getElementById('model');
const status = document.getElementById('status');
const results = document.getElementById('results');
let activeRequest;

for (const input of form.querySelectorAll('input[type=file]')) {
  const identity = document.getElementById(input.getAttribute('aria-describedby'));
  let lookup;
  input.addEventListener('change', async () => {
    lookup?.abort();
    identity.textContent = '';
    const file = input.files[0];
    if (!file?.name.toLowerCase().endsWith('.zip')) return;
    if (file.size > 5 * 1024 ** 2) {
      identity.textContent = 'ZIP exceeds the 5 MB upload limit.';
      return;
    }
    const request = new AbortController();
    lookup = request;
    const fields = new FormData();
    fields.set('first', file);
    identity.textContent = 'Reading Letterboxd username…';
    try {
      const response = await fetch('/api/import-profile', {method: 'POST', body: fields, signal: request.signal});
      const data = await response.json();
      if (request.signal.aborted) return;
      if (!response.ok) throw new Error(data.error);
      identity.textContent = data.username ? `@${data.username}` : 'Username unavailable in this ZIP.';
    } catch {
      if (!request.signal.aborted) identity.textContent = 'Could not read the Letterboxd username.';
    }
  });
}

async function findFilms() {
  activeRequest?.abort();
  const request = new AbortController();
  activeRequest = request;
  const fields = new FormData(form);
  const mode = fields.get('mode');
  if (mode === 'first') fields.delete('second');
  if (mode === 'second') fields.delete('first');

  status.textContent = `Scoring with ${model.selectedOptions[0].textContent}…`;
  results.replaceChildren();
  results.setAttribute('aria-busy', 'true');
  const timeout = setTimeout(() => request.abort('timeout'), 30000);
  try {
    const response = await fetch(form.action, {method: 'POST', body: fields, signal: request.signal});
    const page = new DOMParser().parseFromString(await response.text(), 'text/html');
    if (activeRequest !== request) return;
    const responseStatus = page.getElementById('status');
    const responseResults = page.getElementById('results');
    if (!responseStatus || !responseResults) throw new Error('Invalid server response');
    status.textContent = responseStatus.textContent;
    results.replaceChildren(...responseResults.children);
  } catch {
    if (activeRequest === request) {
      status.textContent = request.signal.reason === 'timeout'
        ? 'Scoring took too long. Select Find films to try again.'
        : 'Could not get recommendations from the local app. Select Find films to retry.';
    }
  } finally {
    clearTimeout(timeout);
    if (activeRequest === request) {
      activeRequest = undefined;
      results.removeAttribute('aria-busy');
    }
  }
}

form.addEventListener('submit', (event) => {
  event.preventDefault();
  findFilms();
});

form.addEventListener('change', (event) => {
  if (event.target !== model && !event.target.matches('input[type="checkbox"][name="seen"]')) return;
  if ((activeRequest || results.childElementCount) && form.reportValidity()) findFilms();
});
