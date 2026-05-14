// Small UI niceties for the PyShiny app.
$(document).on("shiny:value", function(event) {
  if (event.name === "chat_history") {
    setTimeout(function() {
      const chat = document.querySelector(".chat-history");
      if (chat) chat.scrollTop = chat.scrollHeight;
    }, 50);
  }
});
