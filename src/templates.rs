use askama::Template;

#[derive(Template)]
#[template(
    source = r###"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Florence-2 Inference</title>
  <style>
    :root {
      color-scheme: light;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }
    body {
      margin: 0;
      padding: 32px;
    }
    main {
      max-width: 760px;
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #d9dee7;
      border-radius: 8px;
      padding: 28px;
      box-shadow: 0 10px 30px rgba(31, 41, 51, 0.08);
    }
    h1 {
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.2;
    }
    p {
      margin: 0 0 24px;
      color: #52606d;
    }
    form {
      display: grid;
      gap: 18px;
    }
    label,
    legend {
      display: block;
      margin-bottom: 8px;
      font-weight: 650;
      color: #323f4b;
    }
    input[type="file"],
    input[type="text"],
    select,
    textarea {
      box-sizing: border-box;
      width: 100%;
      border: 1px solid #cbd2d9;
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
      background: #ffffff;
    }
    textarea {
      min-height: 92px;
      resize: vertical;
    }
    fieldset {
      margin: 0;
      padding: 0;
      border: 0;
    }
    .radio-row {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }
    .radio-row label {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin: 0;
      font-weight: 500;
    }
    button {
      justify-self: start;
      border: 0;
      border-radius: 6px;
      padding: 10px 16px;
      font: inherit;
      font-weight: 700;
      color: #ffffff;
      background: #2563eb;
      cursor: pointer;
    }
    button:hover {
      background: #1d4ed8;
    }
    .notice {
      margin-top: 22px;
      padding: 14px 16px;
      border-radius: 6px;
      border: 1px solid #b7d7c0;
      background: #edf8f0;
      color: #1f5130;
    }
    .error {
      margin-top: 22px;
      padding: 14px 16px;
      border-radius: 6px;
      border: 1px solid #f3b5b5;
      background: #fff1f1;
      color: #8a1f1f;
    }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.95em;
    }
  </style>
</head>
<body>
  <main>
    <h1>Florence-2 Inference</h1>
    <p>Upload an image, choose a Florence task prompt, and queue inference on the local ONNX worker pool.</p>

    <form method="post" action="/infer-form" enctype="multipart/form-data">
      <div>
        <label for="image">Input Picture</label>
        <input id="image" name="image" type="file" accept="image/*" required>
      </div>

      <fieldset>
        <legend>Task type selector</legend>
        <div class="radio-row">
          <label><input type="radio" name="task_type" value="Single task" checked> Single task</label>
          <label><input type="radio" name="task_type" value="Cascased task"> Cascased task</label>
        </div>
      </fieldset>

      <div>
        <label for="task_prompt">Task Prompt</label>
        <select id="task_prompt" name="task_prompt">
          {% for prompt in single_task_prompts %}
          <option value="{{ prompt }}">{{ prompt }}</option>
          {% endfor %}
        </select>
      </div>

      <div>
        <label for="text_input">Text Input (optional)</label>
        <textarea id="text_input" name="text_input" placeholder="Optional text for grounding, segmentation, region, or open-vocabulary tasks"></textarea>
      </div>

      <button type="submit">Submit</button>
    </form>

    {% if queued %}
    <div class="notice">
      Job queued: <code>{{ job_id }}</code><br>
      Status: <a href="{{ status_url }}">{{ status_url }}</a>
    </div>
    {% endif %}

    {% if error != "" %}
    <div class="error">{{ error }}</div>
    {% endif %}
  </main>

  <script>
    const singlePrompts = [
      {% for prompt in single_task_prompts %}"{{ prompt }}"{% if !loop.last %},{% endif %}{% endfor %}
    ];
    const cascasedPrompts = [
      {% for prompt in cascased_task_prompts %}"{{ prompt }}"{% if !loop.last %},{% endif %}{% endfor %}
    ];
    const taskPrompt = document.getElementById("task_prompt");
    const radios = document.querySelectorAll('input[name="task_type"]');

    function updatePrompts() {
      const taskType = document.querySelector('input[name="task_type"]:checked').value;
      const prompts = taskType === "Cascased task" ? cascasedPrompts : singlePrompts;
      taskPrompt.replaceChildren(...prompts.map((prompt) => {
        const option = document.createElement("option");
        option.value = prompt;
        option.textContent = prompt;
        return option;
      }));
    }

    radios.forEach((radio) => radio.addEventListener("change", updatePrompts));
  </script>
</body>
</html>
"###,
    ext = "html"
)]
pub struct IndexTemplate<'a> {
    pub single_task_prompts: &'a [&'a str],
    pub cascased_task_prompts: &'a [&'a str],
    pub queued: bool,
    pub job_id: String,
    pub status_url: String,
    pub error: String,
}
