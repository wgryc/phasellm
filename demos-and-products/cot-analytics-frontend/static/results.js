function show_prompt() {
    const response = fetch("get_prompt", {
        method: "GET",
        cache: "no-cache",
        credentials: "same-origin",
        headers: {"Content-Type": "application/json"},
    })
    .then(response=>response.json())
    .then(data=>{ 
		var p = data["prompt"].replace(/(?:\r\n|\r|\n)/g, '<br>');
		var prompt_pre = document.getElementById("prompt-info");
		prompt_pre.innerHTML = p;
    })
}

show_prompt();

function resize_textarea(dom_id) {
    var ta = document.getElementById(dom_id);
    ta.style.height = ta.scrollHeight;
}

function add_box(header, notes, code, css_id) {
    var cot_div = document.getElementById('cot-output');
    var notes_clean = notes.replace(/(?:\r\n|\r|\n)/g, '<br>');

    var code_clean = code.replace("```python", "").replace("```", ""); // This needs to be changed to just deleting the first and last line.

    var new_html = `<div class="cot-output-cell" id="cot-cell-code${css_id}">
        <h3>${header}</h3>
        <p class='notes'>${notes_clean}</p>
        <p><textarea class='code' id='code${css_id}'>${code_clean}</textarea></p>
        <p><button class='run-button' onclick="javascript:run('code${css_id}');">Run Code</button></p>
    </div>`;

    cot_div.innerHTML += new_html;
    resize_textarea(`code${css_id}`);
}

function add_code_output(code_output, div_id, is_error) {
    var cot_div = document.getElementById(div_id);
    var new_html = "";
    if (is_error) {
        var new_html = `<span class='heading-error'>Error</span><div class='code-output-after-run'>${code_output}</div>`;
    } else {
        var new_html = `<span class='heading-code-output'>Code Output</span><div class='code-output-after-run'>${code_output}</div>`;
    }
    cot_div.innerHTML += new_html;
}

for (var i = 1; i <= 7; i++) {
    add_box(`Step #${i}`, COT_DATA[i]['objective'], COT_DATA[i]['code_block'], `_step_${i}`);
}

function run(block_id) {
    var code = document.getElementById(block_id).value;
    data = {"code":code};
    console.log(data);
    const response = fetch("runcode", {
        method: "POST",
        cache: "no-cache",
        credentials: "same-origin",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    })
    .then(response=>response.json())
    .then(data=>{ 
        var response = data['response'];
        var code_output = data['code_output'];
        var is_error = data['is_error'];
        //console.log(response);
        //console.log(code_output);
        if (response === "*No outputs.*") {
            code_output = "*No outputs.*"
        }
        //console.log(is_error);
        add_code_output(code_output, `cot-cell-${block_id}`, is_error);
    })
}