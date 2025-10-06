const axios = require("axios");

module.exports = async function (context, req) {
    try {
        const body = req.body || {};

        // Extract inputs
        const promptText = body.text || "";
        const targetLanguage = body.targetLanguage || "French"; // default
        let fileText = "";

        // Decode Base64 file content if provided
        if (body.file) {
            try {
                fileText = Buffer.from(body.file, "base64").toString("utf8");
            } catch (err) {
                context.res = {
                    status: 400,
                    headers: { "Content-Type": "application/json" },
                    body: { error: "Invalid file encoding. Must be base64." }
                };
                return;
            }
        }

        if (!promptText && !fileText) {
            context.res = {
                status: 400,
                headers: { "Content-Type": "application/json" },
                body: { error: "No text or file provided." }
            };
            return;
        }

        // Combine text and file content
        const combinedText = [promptText, fileText].filter(Boolean).join("\n");

        // Azure OpenAI Completion endpoint
        const endpoint = `${process.env.AZURE_OPENAI_ENDPOINT}openai/deployments/${process.env.AZURE_OPENAI_DEPLOYMENT}/completions?api-version=2023-07-01-preview`;

        const response = await axios.post(
            endpoint,
            {
                prompt: `Translate the following text to ${targetLanguage}:\n${combinedText}`,
                max_tokens: 1000,
                temperature: 0.3
            },
            {
                headers: {
                    "api-key": process.env.AZURE_OPENAI_KEY,
                    "Content-Type": "application/json"
                }
            }
        );

        // Return JSON response
        context.res = {
            status: 200,
            headers: { "Content-Type": "application/json" },
            body: {
                targetLanguage,
                originalText: combinedText,
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
