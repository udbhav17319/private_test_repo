const https = require('https');

module.exports = async function (context, req) {
    context.log('LLM Translation API triggered.');

    try {
        let textToTranslate = '';
        let targetLanguage = 'en'; // default

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

        // Override target language if provided in body
        if (req.body && req.body.lang) {
            targetLanguage = req.body.lang;
        }

        // Prepare request to Azure OpenAI
        const apiKey = process.env.OPENAI_KEY;
        const endpoint = process.env.OPENAI_ENDPOINT; // full completions endpoint
        const deployment = process.env.OPENAI_DEPLOYMENT;

        const body = JSON.stringify({
            model: deployment,
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
                    if (res.statusCode >= 200 && res.statusCode < 300) {
                        try {
                            const json = JSON.parse(data);
                            if (json.choices && json.choices.length > 0) {
                                resolve(json.choices[0].text.trim());
                            } else {
                                reject(new Error('No choices returned from LLM.'));
                            }
                        } catch (err) {
                            reject(new Error(`Invalid JSON response: ${data}`));
                        }
                    } else {
                        reject(new Error(`HTTP ${res.statusCode}: ${data}`));
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
