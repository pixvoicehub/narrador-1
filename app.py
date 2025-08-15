# app.py - VERSÃO CORRIGIDA PARA google-genai E gemini-2.5-pro-preview-tts
import os
import io
import mimetypes
import struct
import logging
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS

# Importações da nova biblioteca
from google import genai
from google.genai import types

# --- Configuração de logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """
    Gera um cabeçalho WAV para os dados de áudio recebidos.
    Código copiado/adaptado do exemplo do Google.
    """
    logger.info(f"Convertendo dados de áudio de {mime_type} para WAV...")
    
    # Valores padrão, podem ser ajustados conforme o mime_type
    bits_per_sample = 16
    sample_rate = 24000 
    # Tenta parsear o mime_type para valores reais (opcional, mas ideal)
    # O exemplo do Google faz isso, mas para simplificar, podemos usar padrões razoáveis.
    # Se quiser implementar o parse completo, use a função do exemplo.

    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1, num_channels,
        sample_rate, byte_rate, block_align, bits_per_sample, b"data", data_size
    )
    logger.info("Conversão para WAV concluída.")
    return header + audio_data

@app.route('/')
def home():
    """Rota para verificar se o serviço está online."""
    logger.info("Endpoint '/' acessado.")
    return "Serviço de Narração no Railway está online e estável! (Usando google-genai)"

@app.route('/api/generate-audio', methods=['POST'])
def generate_audio_endpoint():
    """Endpoint principal que gera o áudio."""
    logger.info("Recebendo solicitação para /api/generate-audio")
    
    # 1. Obter a chave da API do ambiente do Railway
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        error_msg = "ERRO CRÍTICO: GEMINI_API_KEY não encontrada nas variáveis de ambiente."
        logger.error(error_msg)
        return jsonify({"error": "Configuração do servidor incompleta: Chave de API ausente."}), 500

    # 2. Obter e validar os dados da requisição
    data = request.get_json()
    if not data:
        error_msg = "Requisição inválida, corpo JSON ausente."
        logger.warning(error_msg)
        return jsonify({"error": error_msg}), 400

    text_to_narrate = data.get('text')
    voice_name = data.get('voice')

    if not text_to_narrate or not voice_name:
        error_msg = "Os campos 'text' e 'voice' são obrigatórios."
        logger.warning(f"{error_msg} Recebido: text={bool(text_to_narrate)}, voice={bool(voice_name)}")
        return jsonify({"error": error_msg}), 400

    try:
        # 3. Configurar o cliente da API
        logger.info("Configurando o cliente Google GenAI...")
        client = genai.Client(api_key=api_key)

        model_name = "gemini-2.5-flash-preview-tts"
        logger.info(f"Usando o modelo: {model_name}")

        # 4. Preparar o conteúdo e a configuração
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=text_to_narrate)]
            )
        ]

        generate_content_config = types.GenerateContentConfig(
            # Outros parâmetros podem ser adicionados aqui
            # temperature=1.0, 
            response_modalities=["audio"], # Especifica que queremos áudio
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name # Especifica a voz
                    )
                )
            )
        )
        logger.info(f"Configuração de geração criada para voz: '{voice_name}'")

        # 5. Fazer a chamada à API (streaming)
        logger.info("Iniciando chamada à API generate_content_stream...")
        audio_data_chunks = []
        for chunk in client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_content_config
        ):
            # Verifica se o chunk contém dados de áudio
            if (chunk.candidates and 
                chunk.candidates[0].content and 
                chunk.candidates[0].content.parts and
                chunk.candidates[0].content.parts[0].inline_data and 
                chunk.candidates[0].content.parts[0].inline_data.data):
                
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                audio_data_chunks.append(inline_data.data)
                logger.debug(f"Recebido chunk de áudio de {len(inline_data.data)} bytes")

        # 6. Processar os dados recebidos
        if not audio_data_chunks:
             error_msg = "A API respondeu, mas não retornou dados de áudio."
             logger.error(error_msg)
             return jsonify({"error": error_msg}), 500

        full_audio_data = b''.join(audio_data_chunks)
        logger.info(f"Dados de áudio completos recebidos ({len(full_audio_data)} bytes).")

        # 7. Converter para WAV se necessário
        # Assumindo que os dados venham em um formato que precise de cabeçalho WAV
        # O exemplo do Google converte, então vamos fazer o mesmo para garantir compatibilidade.
        mime_type = inline_data.mime_type if 'inline_data' in locals() else "audio/unknown"
        logger.info(f"Tipo MIME do áudio recebido: {mime_type}")
        
        # Se não for WAV, converte. (Ou sempre converte, por segurança)
        # A verificação exata do mime_type pode ser feita aqui.
        # Para simplificar, vamos converter sempre.
        wav_data = convert_to_wav(full_audio_data, mime_type)
        
        # 8. Preparar e enviar a resposta de sucesso
        logger.info("Áudio gerado e convertido com sucesso. Enviando resposta...")
        http_response = make_response(send_file(
            io.BytesIO(wav_data),
            mimetype='audio/wav',
            as_attachment=False
        ))
        http_response.headers['X-Model-Used'] = model_name
        http_response.headers['X-Audio-Source-Mime'] = mime_type
        logger.info("Sucesso: Áudio WAV gerado e enviado ao cliente.")
        return http_response

    except Exception as e:
        # Captura qualquer erro que ocorra na comunicação com a API do Google
        error_message = f"Erro ao contatar a API do Google GenAI: {e}"
        logger.error(f"ERRO CRÍTICO NA API: {error_message}", exc_info=True)
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Iniciando aplicação Flask na porta {port}")
    app.run(host='0.0.0.0', port=port)
