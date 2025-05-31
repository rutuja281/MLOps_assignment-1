# Deploying Sentiment Analysis Chatbot – MLOps Case Study



## 📝 Project Overview

This MLOps case study demonstrates the end-to-end deployment of a **Sentiment Analysis Chatbot** using two approaches:  
1. Hosted on Hugging Face Spaces  
2. Locally hosted with open-source LLMs

We explored the full MLOps pipeline including model selection, interface development with Gradio, GitHub Actions for CI/CD, performance benchmarking, cost monitoring, and privacy/security analysis.

---

## 🔧 Product 1: Hugging Face Hosted Chatbot

**GitHub Repository:** [Git_ChatBot_Sync_Hugging_face](https://github.com/YashMalviya16/Git_ChatBot_Sync_Hugging_face)  
**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/YashMalviya16/Mlops)

### 🔹 Model  
- `zephyr-7b-beta` (Transformer-based LLM)
- Inference latency: ~2 seconds
- Hosted on Hugging Face Spaces

### 🔹 Key Features  
- Sentiment classification: Positive / Negative / Neutral  
- Deployed with CI/CD using GitHub Actions  
- Low latency and high availability

### 🔹 Business Use Cases  
- Customer feedback analysis  
- Social media sentiment tracking  
- Review classification

### 🔹 Cost Estimate  
- Free tier suitable for light usage  
- 1,000 users/day → Approx. **$90–$100/month**

---

## 🖥️ Product 2: Locally Hosted Chatbot

**GitHub Repository:** [Case-study-1](https://github.com/rutuja281/Case-study-1)  
**Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/Rutuja28/New_space)

### 🔹 Model  
- `phi-3-mini-4k-instruct` by Microsoft  
- Decoder-only Transformer  
- Inference using `transformers` with `device_map="auto"`  
- Uses bfloat16 precision to reduce memory footprint

### 🔹 Features  
- Fully local execution  
- Gradio interface with custom buttons  
- Open-source and privacy-preserving

### 🔹 Use Cases  
- Private/offline chatbots  
- Prototyping LLM interfaces  
- AI education and experimentation

---

## ⚙️ MLOps Components

- **CI/CD**: GitHub Actions for automated deployment  
- **Webhook Integration**: Discord webhook for push notifications  
- **Security**: Comparison of privacy trade-offs (cloud vs local)  
- **Performance Testing**: Local model slower without GPU acceleration

---

## 📊 Performance Summary

| Feature                  | Hugging Face Hosted     | Locally Hosted             |
|--------------------------|--------------------------|-----------------------------|
| Latency                 | ~2 seconds               | ~15 minutes                 |
| Ease of Deployment      | Very easy                | Manual setup required       |
| Cost                    | Free (low scale)         | Hardware-dependent          |
| Scalability             | High                     | Low (no orchestration)      |
| Security                | Managed by Hugging Face  | Local data control          |

---

## 🔐 Key Takeaways

- Transformers like `phi-3` are powerful but resource-intensive.
- CI/CD using GitHub Actions enables quick redeployments.
- Gradio accelerates UI development for ML interfaces.
- Webhooks improve DevOps team communication.
- Trade-offs between latency, cost, and privacy are central to deployment choices.

---

## 💡 Future Work

- Dockerize the chatbot for cross-platform execution  
- Implement model quantization for better inference speeds  
- Add analytics to track usage and response quality  
- Deploy with GPU-backed Hugging Face Inference API  
- Use GitHub Actions + Docker + Terraform for reproducible MLOps

---

## 📣 Acknowledgments

This project was completed for the **CS553 MLOps course** at **Worcester Polytechnic Institute**, under the mentorship of **Dr. Randy Paffenroth**.

