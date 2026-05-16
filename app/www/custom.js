// Small UI niceties for the PyShiny app.
(function() {
  function scrollChatHistory(event) {
    if (event.name !== "chat_history") return;

    setTimeout(function() {
      const chat = document.querySelector(".chat-history");
      if (chat) chat.scrollTop = chat.scrollHeight;
    }, 50);
  }

  function bindShinyValueHandler() {
    if (!window.jQuery) {
      window.setTimeout(bindShinyValueHandler, 50);
      return;
    }

    window.jQuery(document).on("shiny:value", function(event) {
      scrollChatHistory(event);
    });
  }

  document.addEventListener("shiny:value", scrollChatHistory);
  bindShinyValueHandler();
})();
