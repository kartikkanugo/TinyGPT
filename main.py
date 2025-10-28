from tiny_gpt import RegexTokenizer, env_display_tinygpt_modules, io_load_text_file

env_display_tinygpt_modules()

raw_text = io_load_text_file("The_Verdict.txt")

rt_obj = RegexTokenizer(raw_text)
vocab_list = rt_obj.split()

print(vocab_list)

vocab = rt_obj.create_vocabulary()


text = """"It's the last he painted, you know," 
       Mrs. Gisburn said with pardonable pride."""
ids = rt_obj.encode(text)
print(ids)


text2 = rt_obj.decode(ids)
print(text2)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)


ids = rt_obj.encode(text)
print(ids)

texto = rt_obj.decode(ids)
print(texto)

from tiny_gpt import TikTokenizer

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
joined_text = " <|endoftext|> ".join((text1, text2))

print("\nJoined Text with <|endoftext|> token:")
print(joined_text)


# Initialize tokenizer
tik_obj = TikTokenizer("gpt2", {"<|endoftext|>"})

# Encode text to token IDs
encoded_ids = tik_obj.encode(joined_text)
print("\nEncoded Token IDs:")
print(encoded_ids)

# Decode token IDs back to text
decoded_text = tik_obj.decode(encoded_ids)
print("\nDecoded Text:")
print(decoded_text)

# Optional: Show token count
print("\nTotal Tokens:", tik_obj.count_tokens(joined_text))
