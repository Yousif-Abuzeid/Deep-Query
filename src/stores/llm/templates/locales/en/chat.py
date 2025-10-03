from string import Template

#### chat prompts ####

#### System ####

system_prompt = Template(
    "\n".join(
        [
            "You are an assistant to generate a response for the user.",
            ""
        ]
    )
)

