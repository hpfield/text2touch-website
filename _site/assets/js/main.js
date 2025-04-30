/* assets/js/main.js */

document.addEventListener('DOMContentLoaded', function() {
  // Reward Function component
  const llmSelector = document.getElementById('llmSelector');
  const codeBlock   = document.getElementById('codeBlock');

  if (llmSelector) {
    llmSelector.addEventListener('change', function() {
      const key = llmSelector.value;
      const fileMap = {
        baseline:          'baseline.py',
        gpt4o:             'gpt-4o.py',
        'gemini1.5flash':  'gemini-1.5-flash.py',
        'deepseekr1-671b': 'deepseek-r1-671B.py',
        'llama-3.1-405B':  'llama-3.1-405B.py',
        'o3-mini':         'o3-mini.py'
      };
      const basePath = window.BASEURL || '';  // Jekyll baseurl if set

      const filename = fileMap[key];
      if (!filename) {
        codeBlock.textContent = '# No code available for selected LLM.';
        hljs.highlightElement(codeBlock);
        return;
      }

      const url = `${basePath}/assets/reward_functions/${filename}`;
      console.log('Loading reward fn from:', url);
      fetch(url)
        .then(res => {
          if (!res.ok) throw new Error('File not found');
          return res.text();
        })
        .then(text => {
          codeBlock.textContent = text;
          hljs.highlightElement(codeBlock);
        })
        .catch(() => {
          codeBlock.textContent = '# Error loading reward function.';
          hljs.highlightElement(codeBlock);
        });
    });
    llmSelector.dispatchEvent(new Event('change'));
  }

  // Policy Videos component
  const orientationSelector = document.getElementById('orientationSelector');
  const objectSelector      = document.getElementById('objectSelector');
  const axisSelector        = document.getElementById('axisSelector');
  const videoDebug          = document.getElementById('videoDebug');

  const approachMap = {
    baseline:         'anyrotate',
    gpt4o:            'gpt',
    'gemini1.5flash': 'gemini',
    'deepseekr1-671b':'deepseek'
  };

  const videoElements = {
    baseline:         document.getElementById('video-baseline'),
    gpt4o:            document.getElementById('video-gpt4o'),
    'gemini1.5flash': document.getElementById('video-gemini'),
    'deepseekr1-671b':document.getElementById('video-deepseek')
  };

  function updateVideoSources() {
    const orientation = orientationSelector.value;
    const object      = objectSelector.value;
    const axis        = axisSelector.value;

    let debugText = ''; // Debugging text removed from display

    for (const [key, videoEl] of Object.entries(videoElements)) {
      const prefix   = approachMap[key];
      const filename = `${prefix}_${object}_${axis}_${orientation}.mp4`;
      const relUrl   = `assets/videos/${filename}`;

      // swap in the new source and reload
      const srcEl = videoEl.querySelector('source');
      srcEl.src = relUrl;
      videoEl.load();
      videoEl.autoplay = true; // Enable autoplay
      videoEl.muted = true;    // Mute the video

      // Remove per-video URL display
      const urlDisplay = document.getElementById(`url-${key}`);
      if (urlDisplay) urlDisplay.innerHTML = '';
    }

    if (videoDebug) videoDebug.textContent = ''; // Clear debug box
  }

  orientationSelector.addEventListener('change', updateVideoSources);
  objectSelector.addEventListener('change',    updateVideoSources);
  axisSelector.addEventListener('change',      updateVideoSources);
  updateVideoSources();

  // Experiment Log component (unchanged) …
  const logSelector = document.getElementById('logSelector');
  const logContent  = document.getElementById('logContent');

  function updateLogContent() {
    const selectedLog = logSelector.value;
    const logFilePath = `assets/logs/${selectedLog}.txt`;

    fetch(logFilePath)
      .then(response => {
        if (!response.ok) throw new Error('Log file not found.');
        return response.text();
      })
      .then(text => {
        logContent.textContent = text;
      })
      .catch(() => {
        logContent.textContent = 'Error loading log file.';
      });
  }

  if (logSelector) {
    logSelector.addEventListener('change', updateLogContent);
    updateLogContent();
  }

  // Force all videos except the project video to stay silent
  document.querySelectorAll('video:not(#video-project)').forEach(video => {
    video.muted = true;
    video.volume = 0;
    video.addEventListener('volumechange', () => {
      video.muted = true;
      video.volume = 0;
    });
  });

  // Ensure project video has sound enabled
  const projVideo = document.getElementById('video-project');
  if (projVideo) {
    projVideo.muted = false;
    projVideo.volume = 1;
  }
});
