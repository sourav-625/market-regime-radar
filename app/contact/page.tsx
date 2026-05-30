export default function Contact() {
  return (
    <div className="max-w-4xl mx-auto bg-white p-10 shadow-lg mt-10">

      <h1 className="text-3xl font-bold mb-4">
        Contact Us
      </h1>

      <p className="text-gray-700 mb-6">
        Feel free to reach out to us using the form below.
      </p>

      <form className="space-y-4">

        <input
          type="text"
          placeholder="Your Name"
          className="w-full border p-2 rounded"
        />

        <input
          type="email"
          placeholder="Your Email"
          className="w-full border p-2 rounded"
        />

        <textarea
          placeholder="Your Message"
          className="w-full border p-2 rounded h-32"
        />

        <button className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
          Send Message
        </button>

      </form>

    </div>
  );
}