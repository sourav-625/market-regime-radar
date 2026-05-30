export default function Footer() {
  return (
    <footer className="w-full bg-gray-900 text-gray-300 mt-20">
      
      <div className="max-w-6xl mx-auto px-6 py-10 grid md:grid-cols-3 gap-8">

        {/* Brand */}
        <div>
          <h2 className="text-xl font-semibold text-white mb-2">
            MySite
          </h2>
          <p className="text-sm">
            A modern web platform built with Next.js and Tailwind CSS.
          </p>
        </div>

        {/* Navigation */}
        <div>
          <h3 className="text-white font-medium mb-3">
            Navigation
          </h3>
          <ul className="space-y-2 text-sm">
            <li><a href="#" className="hover:text-white">Home</a></li>
            <li><a href="#" className="hover:text-white">About</a></li>
            <li><a href="#" className="hover:text-white">Contact</a></li>
          </ul>
        </div>

        {/* Info */}
        <div>
          <h3 className="text-white font-medium mb-3">
            Resources
          </h3>
          <ul className="space-y-2 text-sm">
            <li><a href="#" className="hover:text-white">Privacy Policy</a></li>
            <li><a href="#" className="hover:text-white">Terms of Service</a></li>
            <li><a href="#" className="hover:text-white">Support</a></li>
          </ul>
        </div>

      </div>

      <div className="border-t border-gray-700 text-center text-sm py-4">
        © 2026 MySite. All rights reserved.
      </div>

    </footer>
  );
}