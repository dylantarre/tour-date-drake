# Tour Date Drake - SaaS Product Design

## Overview

Tour Date Drake is a dragon-branded tour date formatting tool that serves as a lead generation funnel for LG Media (self-serve ad platform for music sites).

**Primary goal:** Capture emails of music journalists and PR professionals who are ideal LG Media ad buyers.

**Secondary goal:** Provide genuine utility - format messy tour dates into clean, consistent output.

---

## User Tiers

### Anonymous (Teaser)
- Can use the tool without signup
- Sees first 3-5 dates only
- Prompted to enter email for full results
- Rate limit: 3 requests/hour per IP

### Free (Email-gated)
- Unlimited web UI usage
- Full results every time
- 30-day history saved
- Rate limit: 60 requests/hour
- Collects: email (required), name (optional), role (journalist/PR/artist-manager/other)

### API Access
- Not self-serve
- Contact form sends inquiry to admin@lg.media
- Collects: name, email, usage request (what they want API for)
- Manual follow-up and onboarding

---

## Tech Stack

### Application
- **Framework:** FastAPI (extend existing codebase)
- **Database:** PostgreSQL (Supabase free tier or Railway)
- **Hosting:** Single Docker container on Railway/Render/Fly (~$5-10/mo)

### Email Services
- **Mailgun:** Transactional emails (magic links, welcome email, API inquiry confirmation)
- **Loop:** Marketing emails (nurture sequences, LG Media pitches)

### Auth
- Magic link only (no passwords)
- Session cookies: HttpOnly, Secure, SameSite=Lax, 30-day expiry
- Magic link tokens: single-use, 15-minute expiry, hashed in DB

---

## Database Schema

```sql
users
├── id (uuid, primary key)
├── email (unique, indexed)
├── name (nullable)
├── role (enum: journalist, pr, artist_manager, other)
├── loop_subscriber_id (for Loop sync)
├── created_at
└── last_login_at

sessions
├── id (uuid, primary key)
├── user_id (foreign key)
├── token_hash (for magic link validation)
├── expires_at
├── created_at
└── used_at (nullable, set when token consumed)

conversions
├── id (uuid, primary key)
├── user_id (foreign key, nullable for teasers)
├── input_type (enum: text, image_url, image_upload)
├── input_preview (first 200 chars or image thumbnail URL)
├── output_text (full formatted result)
├── model_used
├── created_at
└── is_teaser (boolean - was this a pre-signup preview)

api_inquiries
├── id (uuid, primary key)
├── name
├── email
├── usage_request (text - what they want API for)
├── created_at
└── status (enum: new, contacted, converted, declined)
```

**Notes:**
- Teaser conversions stored with `user_id = null`, linked after signup by session
- History displays last 30 days (query limit, not schema)
- Data auto-deletes after 90 days for privacy

---

## User Flows

### First-time User (Teaser → Signup)
```
1. Visit landing page
2. Paste tour dates or image URL
3. Click "Format Dates"
4. See teaser result (first 3-5 dates)
5. Prompt: "Enter email to unlock full results"
6. Enter email
7. Magic link sent via Mailgun
8. User added to Loop with tags
9. Click magic link
10. Full results displayed
11. Session cookie set (30 days)
```

### Returning User
```
1. Visit site (session cookie present)
2. Automatically logged in
3. Full access to converter + history
```

### Magic Link (Existing User)
```
1. Enter email on login form
2. System checks if user exists (creates if not)
3. Magic link sent
4. No indication whether account existed (security)
```

### API Inquiry
```
1. Click "Need API access? Contact us"
2. Fill form: name, email, usage request
3. Confirmation email via Mailgun
4. Inquiry email to admin@lg.media
5. Manual follow-up
```

---

## Email Strategy

### Mailgun (Transactional)

**Magic Link Email**
- Subject: "Sign in to Tour Date Drake"
- Single CTA button
- 15-minute expiry
- From: drake@tourdatedrake.com (or lg.media subdomain)

**Welcome Email** (after first login)
- Subject: "Welcome to the dragon's lair"
- Quick tips on usage
- Soft mention: "Built by the team behind LG Media"

**API Inquiry Confirmation**
- Subject: "We got your API request"
- Sets expectation: response within 48 hours

### Loop (Marketing)

**On signup, sync to Loop:**
- Email, name, role, signup date
- Tag: `tour-date-drake-user`

**Nurture Sequence:**
- Day 3: "Getting the most out of Tour Date Drake" (tips)
- Day 7: "Promoting a tour? Reach 2M+ music fans" (lg.media intro)
- Day 14: "LG Media: ads across Lambgoat, Needle Drop, No Echo & more" (direct pitch)

---

## UI/UX Design

### Branding
- **Mascot:** Dragon (Spyro meets productivity tool)
- **Tagline:** "Tame your tour dates"
- **Colors:** Deep purple (#4A1D6A), burnt orange (#E85D04), cream (#FDF6E3)
- **Font:** Inter or Space Grotesk
- **Vibe:** Playful but professional

### Dragon Touches
- Subtle scale texture on cards
- Fire/smoke loading animation while processing
- Dragon tail decorative element
- Egg icon for empty states

### Pages

**Landing (Anonymous)**
- Logo + tagline
- Converter UI (text/image URL tabs)
- Social proof: "Used by writers at..."

**Teaser Result**
- First 3-5 dates shown
- "12 more dates" locked indicator
- Email capture form
- "Free forever. No credit card."

**Dashboard (Logged In)**
- Same converter UI
- History sidebar (recent conversions)
- "Need API access? Contact us" link

**Other Pages**
- Magic link sent / check inbox
- API inquiry form
- Sign out confirmation

---

## Security & Edge Cases

### Rate Limiting
- Anonymous: 3 requests/hour per IP
- Logged in: 60 requests/hour per user

### Input Validation
- Text: max 50,000 characters
- Image URL: validate format, fetch timeout, max 20MB
- Image upload: max 20MB, validate MIME type

### Edge Cases

| Scenario | Handling |
|----------|----------|
| Email already exists | Send magic link (no "account exists" leak) |
| Magic link clicked twice | First works, second shows "link expired" |
| AI returns garbage | Show error, don't save, log for debugging |
| No dates found | Friendly message with retry suggestion |
| Mailgun fails | Queue retry, show "check inbox" anyway |
| Loop API fails | Log error, don't block signup |

### Privacy
- History auto-deletes after 90 days
- "Delete my account" option: wipes user + history + unsubscribes Loop

---

## Not Building (YAGNI)

- Self-serve API signups / Stripe payments
- Admin dashboard (manage via DB directly)
- Password authentication
- OAuth (Google, GitHub, etc.)
- Team/organization accounts
- Usage analytics dashboard

---

## Success Metrics

1. **Email capture rate:** % of teaser users who sign up
2. **Loop → LG Media conversion:** users who click through to lg.media
3. **Ad sales attributed:** LG Media purchases from Tour Date Drake leads
4. **Retention:** returning users per week

---

## Implementation Priority

1. Database schema + migrations
2. Magic link auth flow (Mailgun)
3. Teaser → full result gating
4. User history storage/display
5. Loop integration (signup sync)
6. API inquiry form
7. UI polish + dragon branding
8. Email templates + nurture sequence (in Loop)
