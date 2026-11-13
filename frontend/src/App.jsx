
import { useState, useRef, useEffect } from "react";

import {
  MessageSquare,
  Plus,
  Search,
  Paperclip,
  Send,
  Menu,
  PanelLeftClose,
  Bot,
  Sparkles,
  Trash2,
  FileText,
  ChevronDown,
  LoaderCircle,
  X,
} from "lucide-react";

import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const ACCEPTED_FILES = ".pdf,.png,.jpg,.jpeg,.webp";
const MAX_FILE_SIZE = 15 * 1024 * 1024;

function App() {

  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);

  const [input, setInput] = useState("");
  const [selectedFiles, setSelectedFiles] = useState([]);

  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [error, setError] = useState("");

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const currentChat = chats.find(
    (chat) => chat.id === activeChat
  );

  const messages = currentChat?.messages || [];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  }, [messages, loading]);

  // CREATE NEW CHAT

  function createNewChat() {

    const id = Date.now();

    setChats((previous) => [
      {
        id,
        title: "New chat",
        messages: [],
      },
      ...previous,
    ]);

    setActiveChat(id);
    setInput("");
    setSelectedFiles([]);
    setError("");
  }

  // SELECT CHAT

  function selectChat(id) {
    setActiveChat(id);
    setError("");
    setSelectedFiles([]);
  }

  // DELETE CHAT

  function deleteChat(id) {

    const updated = chats.filter(
      (chat) => chat.id !== id
    );

    setChats(updated);

    if (activeChat === id) {
      setActiveChat(updated[0]?.id || null);
    }
  }

  // FILE SELECTION

  function handleFileSelection(event) {

    const incoming = Array.from(
      event.target.files || []
    );

    const allowed = [];
    const rejected = [];

    incoming.forEach((file) => {

      const extension = file.name
        .split(".")
        .pop()
        ?.toLowerCase();

      if (
        !["pdf", "png", "jpg", "jpeg", "webp"].includes(extension)
      ) {
        rejected.push(
          `${file.name}: unsupported file type`
        );
      } else if (file.size > MAX_FILE_SIZE) {
        rejected.push(
          `${file.name}: exceeds 15 MB`
        );
      } else {
        allowed.push(file);
      }

    });

    setSelectedFiles((previous) => {

      const combined = [...previous];

      allowed.forEach((file) => {

        const duplicate = combined.some(
          (item) =>
            item.name === file.name &&
            item.size === file.size &&
            item.lastModified === file.lastModified
        );

        if (!duplicate) {
          combined.push(file);
        }

      });

      return combined;
    });

    setError(rejected.join(". "));

    // Allow selecting the same file again.
    event.target.value = "";
  }

  // REMOVE SELECTED FILE

  function removeSelectedFile(indexToRemove) {

    setSelectedFiles((previous) =>
      previous.filter(
        (_, index) => index !== indexToRemove
      )
    );
  }

  // SEND MESSAGE AND FILES

  async function sendMessage(event) {

    event.preventDefault();

    const question = input.trim();

    const filesToSend = [...selectedFiles];

    if (
      (!question && filesToSend.length === 0) ||
      loading
    ) {
      return;
    }

    const prompt =
      question || "Please analyze the attached file(s).";

    let chatId = activeChat;

    if (!chatId) {

      chatId = Date.now();

      setActiveChat(chatId);

      setChats((previous) => [
        {
          id: chatId,
          title: prompt.slice(0, 35),
          messages: [],
        },
        ...previous,
      ]);
    }

    const userMessage = {

      role: "user",

      content: prompt,

      attachments: filesToSend.map((file) => ({
        name: file.name,
        size: file.size,
      })),

    };

    setChats((previous) =>
      previous.map((chat) =>

        chat.id === chatId
          ? {
              ...chat,

              title:
                chat.messages.length === 0
                  ? prompt.slice(0, 35)
                  : chat.title,

              messages: [
                ...chat.messages,
                userMessage,
              ],
            }
          : chat

      )
    );

    setInput("");
    setSelectedFiles([]);
    setLoading(true);
    setError("");

    try {

      let response;

      // IF FILES ARE ATTACHED
      if (filesToSend.length > 0) {

        const formData = new FormData();

        formData.append("message", prompt);

        filesToSend.forEach((file) => {
          formData.append("files", file);
        });

        response = await fetch(
          `${API_URL}/api/chat/upload`,
          {
            method: "POST",
            body: formData,
          }
        );

      } else {

        // NORMAL TEXT CHAT

        response = await fetch(
          `${API_URL}/api/chat`,
          {
            method: "POST",

            headers: {
              "Content-Type": "application/json",
            },

            body: JSON.stringify({
              message: prompt,
            }),
          }
        );

      }

      const responseText = await response.text();

      let data = {};

      try {
        data = responseText
          ? JSON.parse(responseText)
          : {};
      } catch {
        throw new Error(
          responseText ||
          `Server returned HTTP ${response.status}`
        );
      }

      if (!response.ok) {

        const detail =
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail || data);

        throw new Error(
          detail || `Request failed (${response.status})`
        );
      }

      const assistantMessage = {

        role: "assistant",

        content:
          data.answer ||
          data.response ||
          data.message ||
          "The server returned no answer text.",

        sources: data.sources || [],

        model: data.model,

      };

      setChats((previous) =>
        previous.map((chat) =>

          chat.id === chatId
            ? {
                ...chat,

                messages: [
                  ...chat.messages,
                  assistantMessage,
                ],
              }
            : chat

        )
      );

    } catch (err) {

      setError(
        `${err.message}. Check that the FastAPI server is running at ${API_URL}.`
      );

    } finally {

      setLoading(false);

    }
  }

  return (

    <div className="app-container">

      {/* SIDEBAR */}

      <aside
        className={`sidebar ${
          sidebarOpen ? "" : "sidebar-hidden"
        }`}
      >

        <div className="sidebar-top">

          <div className="brand">

            <div className="brand-icon">
              <Bot size={23} />
            </div>

            <span>My AI</span>

          </div>

          <button
            className="icon-button"
            onClick={() => setSidebarOpen(false)}
            title="Close sidebar"
          >
            <PanelLeftClose size={19} />
          </button>

        </div>

        <button
          className="new-chat-button"
          onClick={createNewChat}
        >
          <Plus size={19} />
          <span>New chat</span>
        </button>

        <div className="sidebar-search">
          <Search size={17} />
          <span>Search chats</span>
        </div>

        <div className="history-heading">
          Your conversations
        </div>

        <div className="chat-history">

          {chats.map((chat) => (

            <div
              key={chat.id}
              className={`history-item ${
                activeChat === chat.id
                  ? "history-active"
                  : ""
              }`}
            >

              <button
                className="history-select"
                onClick={() => selectChat(chat.id)}
              >

                <MessageSquare size={16} />

                <span>{chat.title}</span>

              </button>

              <button
                className="delete-button"
                onClick={() => deleteChat(chat.id)}
                title="Delete chat"
              >
                <Trash2 size={15} />
              </button>

            </div>

          ))}

          {chats.length === 0 && (

            <p className="empty-history">
              Your conversations will appear here.
            </p>

          )}

        </div>

        <div className="sidebar-footer">

          <div className="profile-icon">
            S
          </div>

          <div className="profile-details">

            <strong>Personal Assistant</strong>

            <span>Local AI workspace</span>

          </div>

        </div>

      </aside>

      {/* MAIN CONTENT */}

      <main className="main-content">

        {/* HEADER */}

        <header className="topbar">

          <div className="header-left">

            {!sidebarOpen && (

              <button
                className="icon-button"
                onClick={() => setSidebarOpen(true)}
              >
                <Menu size={21} />
              </button>

            )}

            <div className="model-selector">

              <span>My AI Assistant</span>

              <ChevronDown size={16} />

            </div>

          </div>

          <div className="header-right">

            <span className="local-status">

              <span className="status-dot" />

              Local AI

            </span>

          </div>

        </header>

        {/* CHAT BODY */}

        <section className="chat-area">

          {messages.length === 0 ? (

            <div className="welcome-screen">

              <div className="welcome-icon">
                <Sparkles size={30} />
              </div>

              <h1>
                How can I help you today?
              </h1>

              <p>
                Ask anything. Explore your documents.
                Let's get started.
              </p>

              <div className="suggestion-grid">

                <button
                  onClick={() =>
                    setInput(
                      "What are the main topics covered in Unit 1?"
                    )
                  }
                >
                  <FileText size={20} />
                  <strong>Analyze documents</strong>
                  <span>
                    Ask questions about your PDFs and images
                  </span>
                </button>

                <button
                  onClick={() =>
                    setInput(
                      "Explain artificial intelligence in simple language."
                    )
                  }
                >
                  <Sparkles size={20} />
                  <strong>Learn something</strong>
                  <span>
                    Understand complex concepts easily
                  </span>
                </button>

                <button
                  onClick={() =>
                    setInput(
                      "Explain how large language models work."
                    )
                  }
                >
                  <Bot size={20} />
                  <strong>Explore AI</strong>
                  <span>
                    Understand artificial intelligence
                  </span>
                </button>

                <button
                  onClick={() =>
                    setInput(
                      "Summarize the uploaded documents."
                    )
                  }
                >
                  <MessageSquare size={20} />
                  <strong>Summarize</strong>
                  <span>
                    Get quick document summaries
                  </span>
                </button>

              </div>

            </div>

          ) : (

            <div className="messages-container">

              {messages.map((message, index) => (

                <div
                  key={index}
                  className={`message-row ${
                    message.role === "user"
                      ? "user-row"
                      : "assistant-row"
                  }`}
                >

                  {message.role === "assistant" && (

                    <div className="message-avatar">
                      <Bot size={19} />
                    </div>

                  )}

                  <div className="message-content">

                    <div className="message-role">

                      {message.role === "user"
                        ? "You"
                        : "My AI"}

                    </div>

                    <div className="message-text">
                      {message.content}
                    </div>

                    {message.attachments?.length > 0 && (

                      <div className="message-attachments">

                        {message.attachments.map(
                          (file, fileIndex) => (

                            <div
                              className="message-attachment"
                              key={`${file.name}-${fileIndex}`}
                            >
                              <FileText size={15} />
                              {file.name}
                            </div>

                          )
                        )}

                      </div>

                    )}

                    {message.sources?.length > 0 && (

                      <div className="sources-box">

                        <strong>
                          <FileText size={15} />
                          Sources
                        </strong>

                        {message.sources.map(
                          (source, sourceIndex) => (

                            <div key={sourceIndex}>
                              {source.source} · Page/Slide{" "}
                              {source.page}
                            </div>

                          )
                        )}

                      </div>

                    )}

                    {message.model && (

                      <span className="model-label">
                        {message.model}
                      </span>

                    )}

                  </div>

                </div>

              ))}

              {loading && (

                <div className="loading-message">

                  <LoaderCircle
                    size={19}
                    className="spinner"
                  />

                  My AI is thinking...

                </div>

              )}

              <div ref={messagesEndRef} />

            </div>

          )}

        </section>

        {/* ERROR MESSAGE */}

        {error && (

          <div className="error-message">

            <span>{error}</span>

            <button onClick={() => setError("")}>
              <X size={16} />
            </button>

          </div>

        )}

        {/* INPUT AREA */}

        <footer className="input-area">

          <form
            className="chat-input-box"
            onSubmit={sendMessage}
          >

            {/* HIDDEN FILE INPUT */}

            <input
              ref={fileInputRef}
              className="hidden-file-input"
              type="file"
              accept={ACCEPTED_FILES}
              multiple
              onChange={handleFileSelection}
            />

            {/* ATTACHMENT BUTTON */}

            <button
              type="button"
              className="input-icon"
              title="Attach PDF or image"
              onClick={() =>
                fileInputRef.current?.click()
              }
              disabled={loading}
            >
              <Paperclip size={21} />
            </button>

            <div className="composer-main">

              {/* SELECTED FILE PREVIEW */}

              {selectedFiles.length > 0 && (

                <div className="selected-files">

                  {selectedFiles.map((file, index) => (

                    <div
                      className="selected-file"
                      key={`${file.name}-${file.size}-${file.lastModified}`}
                    >

                      <FileText size={15} />

                      <span title={file.name}>
                        {file.name}
                      </span>

                      <button
                        type="button"
                        onClick={() =>
                          removeSelectedFile(index)
                        }
                        aria-label={`Remove ${file.name}`}
                      >
                        <X size={14} />
                      </button>

                    </div>

                  ))}

                </div>

              )}

              {/* MESSAGE INPUT */}

              <textarea
                value={input}
                onChange={(event) =>
                  setInput(event.target.value)
                }
                placeholder="Message My AI Assistant..."
                rows={1}
                onKeyDown={(event) => {

                  if (
                    event.key === "Enter" &&
                    !event.shiftKey
                  ) {

                    event.preventDefault();

                    // Submit via the form so onSubmit fires
                    // exactly once (avoids double-submit).
                    event.target.form?.requestSubmit();

                  }

                }}
              />

            </div>

            {/* SEND BUTTON */}

            <button
              type="submit"
              className="send-button"
              disabled={
                (!input.trim() &&
                  selectedFiles.length === 0) ||
                loading
              }
              title="Send message"
            >

              {loading ? (
                <LoaderCircle
                  size={19}
                  className="spinner"
                />
              ) : (
                <Send size={19} />
              )}

            </button>

          </form>

          <p className="disclaimer">
            PDF, PNG, JPG, JPEG and WEBP · Maximum 15 MB per file
          </p>

          <p className="disclaimer">
            My AI can make mistakes. Verify important information.
          </p>

        </footer>

      </main>

    </div>

  );
}

export default App;
