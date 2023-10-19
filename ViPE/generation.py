
def generate_from_sentences(text, model, tokenizer,device,do_sample,top_k=100, epsilon_cutoff=.00005, temperature=1):
    text=[tokenizer.eos_token +  i + tokenizer.eos_token for i in text]
    batch=tokenizer(text, padding=True, return_tensors="pt")

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    max_prompt_length=50

    generated_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask, max_new_tokens=max_prompt_length, do_sample=do_sample,top_k=top_k, epsilon_cutoff=epsilon_cutoff, temperature=temperature)
    pred_caps = tokenizer.batch_decode(generated_ids[:, -(generated_ids.shape[1] - input_ids.shape[1]):], skip_special_tokens=True)


    return pred_caps