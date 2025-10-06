const fs = require("fs");
const path = require("path");
const multiparty = require("multiparty");
const axios = require("axios");

module.exports = async function (context, req) {
    try {
        if (req.method === "GET") {
            // Serve HTML page
            const filePath = path.join(__dirname, "..", "static", "index.html");
            const html = fs.readFileSync(filePath, "utf8");
            context.res = {
                status: 200,
                headers: { "Content-Type": "text/html" },
                body: html
            };
            return;
        }

        // Handle POST (text or file)
        const form = new multiparty.Form();
        const data = await new Promise((resolve, reject) => {
            form.parse(req, (err, fields, files) => {
                if (err) reject(err);
                else resolve({ fields, files });
            });
        });

        let promptText = "";

        // If file uploaded, read content
        if (data.files && data.files.file && data.files.file[0]) {
            const fileContent = fs.readFileSync(data.files.file[0].path, "utf8");
            promptText += fileContent;
        }

        // If text provided
        if (data.fields && data.fields.text) {
            promptText += (promptText ? "\n" : "") + data.fields.text[0];
        }

        if (!promptText) {
            context.res = {
                status: 400,
                headers: { "Content-Type": "application/json" },
                body: { error: "No file or text provided." }
            };
            return;
        }

        // Call Azure OpenAI Completion Endpoint
        const endpoint = `${process.env.AZURE_OPENAI_ENDPOINT}openai/deployments/${process.env.AZURE_OPENAI_DEPLOYMENT}/completions?api-version=2023-07-01-preview`;

        const response = await axios.post(endpoint, {
            prompt: `Translate this text to French:\n${promptText}`,
            max_tokens: 500,
            temperature: 0.7
        }, {
            headers: {
                "api-key": process.env.AZURE_OPENAI_KEY,
                "Content-Type": "application/json"
            }
        });

        context.res = {
            status: 200,
            headers: { "Content-Type": "application/json" },
            body: {
                prompt: promptText,
                translation: response.data.choices[0].text
            }
        };

    } catch (err) {
        context.res = {
            status: 500,
            headers: { "Content-Type": "application/json" },
            body: { error: err.message }
        };
    }
};
