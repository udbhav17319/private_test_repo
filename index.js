const https = require('https');

module.exports = async function (context, req) {
    context.log('LLM Translation API triggered.');

    try {
        let textToTranslate = '';
        const targetLanguage = req.query.lang || 'en';

        // Handling text input
        if (req.body && req.body.text) {
            textToTranslate = req.body.text;
        }
        // Handling file input (base64-encoded text file)
        else if (req.body && req.body.file) {
            const fileBuffer = Buffer.from(req.body.file, 'base64');
            textToTranslate = fileBuffer.toString('utf8');
        } else {
            context.res = { status: 400, body: 'Provide text or file in request body.' };
            return;
        }

        // Prepare request to Azure OpenAI / OpenAI
        const apiKey = process.env.OPENAI_KEY;
        const endpoint = process.env.OPENAI_ENDPOINT; // e.g., https://<your-resource>.openai.azure.com/openai/deployments/<deployment>/completions?api-version=2023-07-01-preview

        const body = JSON.stringify({
            model: process.env.OPENAI_DEPLOYMENT,  // deployment name
            prompt: `Translate the following text to ${targetLanguage}:\n\n${textToTranslate}`,
            max_tokens: 1000,
            temperature: 0
        });

        const options = {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'api-key': apiKey,
                'Content-Length': Buffer.byteLength(body)
            }
        };

        const translation = await new Promise((resolve, reject) => {
            const reqApi = https.request(endpoint, options, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    try {
                        const json = JSON.parse(data);
                        // Azure OpenAI: response may be in json.choices[0].text
                        resolve(json.choices[0].text.trim());
                    } catch (err) {
                        reject(err);
                    }
                });
            });

            reqApi.on('error', reject);
            reqApi.write(body);
            reqApi.end();
        });

        context.res = {
            status: 200,
            body: { translation }
        };

    } catch (error) {
        context.res = {
            status: 500,
            body: { error: error.message }
        };
    }
};
